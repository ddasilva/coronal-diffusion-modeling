import argparse
from datetime import datetime
import os
from pathlib import Path

from astropy.io import fits
from matplotlib.dates import date2num
import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import pandas as pd
import pyshtools
from tqdm import tqdm

from coronal_diffusion import constants
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--lat-step", type=float, default=0.5)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/peaklat_testdata",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    train_times, train_file_paths = get_r000_train_file_paths()
    train_times_d2n = date2num(train_times)

    parallel = Parallel(n_jobs=args.n_jobs)
    tasks = [
        delayed(write_peaklat_file)(
            time=time,
            train_times=train_times,
            train_times_d2n=train_times_d2n,
            train_file_paths=train_file_paths,
            lats=lats,
            lons=lons,
            radius=args.radius,
            out_dir=out_dir,
        )
        for time in date_range
    ]

    with joblib_progress("Writing peak latitude testdata...", total=len(tasks)):
        parallel(tasks)

    print(f"Wrote {len(date_range)} files to {out_dir}")


def get_r000_train_file_paths():
    train_file_paths = os.listdir(config.train_wsa_dir)
    train_file_paths = [f for f in train_file_paths if f.endswith(".fits") and "R000" in f]
    train_file_paths.sort()

    times = []
    for file_path in train_file_paths:
        time = datetime.strptime(
            os.path.basename(file_path).split("R")[0], "wsa_%Y%m%d%H%M"
        )
        times.append(time)

    return np.array(times), np.array(train_file_paths)


def read_ortho_harmonics(file_path):
    with fits.open(file_path) as fits_file:
        sph_data = fits_file[3].data.copy()

    g_coeffs = sph_data[0, :, :].T
    h_coeffs = sph_data[1, :, :].T

    coeffs = pyshtools.SHMagCoeffs.from_array(
        np.array([g_coeffs, h_coeffs]),
        normalization="schmidt",
        r0=1,
    )
    coeffs = coeffs.convert(normalization="ortho", kind='complex', csphase=1)

    real = coeffs.coeffs[0].real
    imag = coeffs.coeffs[0].imag
    
    return real, imag


def write_peaklat_file(
    time, train_times, train_times_d2n, train_file_paths, lats, lons, radius, out_dir
):
    idx = np.argmin(np.abs(train_times_d2n - date2num(time)))
    base_file = os.path.join(config.train_wsa_dir, train_file_paths[idx])

    real_file_path = base_file
    real, imag = read_ortho_harmonics(real_file_path)
    peak_lat_north, peak_lat_south = compute_peak_lat_grid(
        real, imag, lats=lats, lons=lons, radius=radius
    )

    timestamp = time.isoformat()
    file_path = out_dir / f"{timestamp}.hdf5"

    with h5py.File(file_path, "w") as hdf:
        hdf["time_d2n"] = date2num(time)
        hdf["time_str"] = timestamp
        hdf["lats"] = lats
        hdf["radius"] = np.array([radius], dtype=np.float32)
        hdf["peak_lat_north"] = peak_lat_north
        hdf["peak_lat_south"] = peak_lat_south
        hdf.attrs["matched_wsa_time_iso"] = train_times[idx].isoformat()
        hdf.attrs["source_r000_file"] = str(base_file)
        hdf.attrs["peak_lat_north_definition"] = (
            "Latitude (degrees) where the magnetic field strength is maximum in the "
            "northern hemisphere (0° to 90°), maximizing across all longitudes, "
            "at the specified radius."
        )
        hdf.attrs["peak_lat_south_definition"] = (
            "Latitude (degrees) where the magnetic field strength is maximum in the "
            "southern hemisphere (-90° to 0°), maximizing across all longitudes, "
            "at the specified radius."
        )


def compute_peak_lat_grid(real, imag, lats, lons, radius):
    """
    Compute hemisphere peak latitudes at the specified radius by maximizing
    magnetic field strength across all longitudes.

    Parameters
    ----------
    real : ndarray
        Real part of spherical harmonic coefficients
    imag : ndarray
        Imaginary part of spherical harmonic coefficients
    lats : ndarray
        Latitude grid (degrees)
    lons : ndarray
        Longitude grid (degrees)
    radius : float
        Radius at which to evaluate field (solar radii)

    Returns
    -------
    peak_lat_north : float
        Peak latitude in northern hemisphere across all longitudes
    peak_lat_south : float
        Peak latitude in southern hemisphere across all longitudes
    """
    coeffs = pyshtools.SHMagCoeffs.from_array(
        np.array([real, imag]),
        normalization="ortho",
        r0=1,
    )

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    result = coeffs.expand(
        r=np.full(lat_grid.size, radius, dtype=np.float64),
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        degrees=True,
    )

    br, btheta, bphi = result.T
    field_strength = np.sqrt(br**2 + btheta**2 + bphi**2)
    field_strength_grid = field_strength.reshape(lat_grid.shape)

    # Split into northern and southern hemispheres
    # Northern: lats >= 0
    # Southern: lats < 0
    lat_north_mask = lats >= 0
    lat_south_mask = lats < 0

    if lat_north_mask.sum() > 0:
        north_strength = field_strength_grid[lat_north_mask, :]
        north_flat_idx = np.argmax(north_strength)
        north_lat_idx, _ = np.unravel_index(north_flat_idx, north_strength.shape)
        peak_lat_north = np.float32(lats[lat_north_mask][north_lat_idx])
    else:
        peak_lat_north = np.float32(np.nan)

    if lat_south_mask.sum() > 0:
        south_strength = field_strength_grid[lat_south_mask, :]
        south_flat_idx = np.argmax(south_strength)
        south_lat_idx, _ = np.unravel_index(south_flat_idx, south_strength.shape)
        peak_lat_south = np.float32(lats[lat_south_mask][south_lat_idx])
    else:
        peak_lat_south = np.float32(np.nan)

    return peak_lat_north, peak_lat_south


if __name__ == "__main__":
    main()
