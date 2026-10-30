import argparse
from datetime import datetime
import os
from pathlib import Path

from astropy.io import fits
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
import pyshtools

from coronal_diffusion import constants
from write_validation_unsignedflux_samples import (
    build_flux_grid,
    integrate_unsigned_flux,
)
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-realizations", type=int, default=1)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
    )
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--wsa-dir", type=str, default=config.train_wsa_dir)
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/validation/unsignedflux_testdata.h5",
    )
    args = parser.parse_args()

    if not 1 <= args.n_realizations <= constants.N_REAL:
        raise ValueError(f"--n-realizations must be in [1, {constants.N_REAL}].")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats, lons, solid_angle = build_flux_grid(args.lat_step, args.lon_step)
    rs = np.array(args.radii, dtype=np.float64)

    if lats.size == 0 or lons.size == 0 or rs.size == 0:
        raise ValueError("Latitude, longitude, and radius grids must be non-empty.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    wsa_times, wsa_file_paths = get_r000_file_paths(args.wsa_dir)
    wsa_times_d2n = date2num(wsa_times)

    # Gather (time index, realization index, file path) items to process.
    iter_items = []
    matched_times = []
    for i, time in enumerate(date_range):
        idx = np.argmin(np.abs(wsa_times_d2n - date2num(time)))
        base_file = os.path.join(args.wsa_dir, wsa_file_paths[idx])
        matched_times.append(wsa_times[idx].isoformat())

        for j in range(args.n_realizations):
            iter_items.append((i, j, base_file.replace("R000", f"R{j:03d}")))

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(process_item)(i, j, file_path, lats, lons, rs, solid_angle)
        for i, j, file_path in iter_items
    ]
    with joblib_progress("Computing unsigned flux...", total=len(tasks)):
        results = parallel(tasks)

    flux = np.full(
        (len(date_range), args.n_realizations, rs.size), np.nan, dtype=np.float64
    )
    for i, j, flux_ij in results:
        flux[i, j] = flux_ij

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(date_range)
        hdf["rs"] = rs
        hdf["unsigned_flux"] = flux
        hdf["matched_wsa_time_iso"] = np.array(matched_times, dtype="S")
        hdf.attrs["source_wsa_dir"] = str(args.wsa_dir)
        hdf.attrs["n_realizations"] = args.n_realizations
        hdf.attrs["lat_step"] = args.lat_step
        hdf.attrs["lon_step"] = args.lon_step
        hdf.attrs["flux_definition"] = (
            "Unsigned radial flux, integral of |Br| r^2 dOmega over the sphere "
            "of each radius r (in solar radii), in units of [Br] * Rs^2. "
            "Axes: (time, realization, radius)."
        )

    print(f"Wrote {len(date_range)} timestamps to {out_file}")


def get_r000_file_paths(wsa_dir):
    file_paths = os.listdir(wsa_dir)
    file_paths = [f for f in file_paths if f.endswith(".fits") and "R000" in f]
    file_paths.sort()

    times = []
    for file_path in file_paths:
        time = datetime.strptime(
            os.path.basename(file_path).split("R")[0], "wsa_%Y%m%d%H%M"
        )
        times.append(time)

    return np.array(times), np.array(file_paths)


def process_item(i, j, file_path, lats, lons, rs, solid_angle):
    return i, j, compute_unsigned_flux(file_path, lats, lons, rs, solid_angle)


def compute_unsigned_flux(file_path, lats, lons, rs, solid_angle):
    from pyshtools.backends.shtools import SHctor

    with fits.open(file_path) as fits_file:
        sph_data = fits_file[3].data.copy()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    coeffs = pyshtools.SHCoeffs.from_array(
        np.array([G, H]),
        normalization="schmidt",
        csphase=-1,
    )
    coeffs = coeffs.convert(normalization="ortho", kind="complex", csphase=-1)
    
    real = coeffs.coeffs[0].real
    imag = coeffs.coeffs[0].imag

    complex_array = np.empty((2,) + real.shape, dtype=np.float64)
    complex_array[0] = real
    complex_array[1] = imag

    real_coeffs = SHctor(complex_array)
    coeffs = pyshtools.SHMagCoeffs.from_array(
        real_coeffs,
        normalization="ortho",
        r0=1,
    )

    lat_grid, lon_grid, r_grid = np.meshgrid(lats, lons, rs, indexing="ij")
    result = coeffs.expand(
        r=r_grid.ravel(),
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        degrees=True,
    )

    # Same rescaling as write_validation_bcube_testdata.py, so the field is on the
    # same scale as the harmonic samples. It matters here because the flux uses |Br|.
    #result /= 4 * np.pi * np.sqrt(2)
    br = result.T[0].reshape(lat_grid.shape)

    return integrate_unsigned_flux(br, solid_angle, rs)


if __name__ == "__main__":
    main()
