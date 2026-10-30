import argparse
from datetime import datetime
import os
from pathlib import Path

from astropy.io import fits
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import contourpy
import h5py
import numpy as np
import pandas as pd
import pyshtools

from coronal_diffusion import constants
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-realizations", default=1, type=int)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/validation/csextent_testdata.h5",
    )
    args = parser.parse_args()

    if not 1 <= args.n_realizations <= constants.N_REAL:
        raise ValueError(f"--n-realizations must be in [1, {constants.N_REAL}].")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    train_times, train_file_paths = get_r000_train_file_paths()
    train_times_d2n = date2num(train_times)

    # Gather (time index, realization index, file path) items to process.
    iter_items = []
    matched_times = []
    for i, time in enumerate(date_range):
        idx = np.argmin(np.abs(train_times_d2n - date2num(time)))
        base_file = os.path.join(config.train_wsa_dir, train_file_paths[idx])
        matched_times.append(train_times[idx].isoformat())

        for j in range(args.n_realizations):
            iter_items.append((i, j, base_file.replace("R000", f"R{j:03d}")))

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(process_item)(
            i, j, file_path, lats, lons, args.radius, args.level
        )
        for i, j, file_path in iter_items
    ]
    with joblib_progress("Computing current sheet extents...", total=len(tasks)):
        results = parallel(tasks)

    lat_min = np.full((len(date_range), args.n_realizations), np.nan, dtype=np.float32)
    lat_max = np.full((len(date_range), args.n_realizations), np.nan, dtype=np.float32)
    for i, j, extent in results:
        lat_min[i, j] = extent[0]
        lat_max[i, j] = extent[1]

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(date_range)
        hdf["lat_min"] = lat_min
        hdf["lat_max"] = lat_max
        hdf["lat_extent"] = lat_max - lat_min
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["radius"] = np.array([args.radius], dtype=np.float32)
        hdf["matched_wsa_time_iso"] = np.array(matched_times, dtype="S")
        hdf.attrs["source_wsa_dir"] = str(config.train_wsa_dir)
        hdf.attrs["n_realizations"] = args.n_realizations
        hdf.attrs["contour_level"] = args.level
        hdf.attrs["extent_definition"] = (
            "Minimum and maximum latitude (degrees) reached by the Br contour at "
            "the given level and radius. NaN where no contour exists."
        )

    print(f"Wrote {len(date_range)} timestamps to {out_file}")


def get_r000_train_file_paths():
    train_file_paths = os.listdir(config.train_wsa_dir)
    train_file_paths = [
        f for f in train_file_paths if f.endswith(".fits") and "R000" in f
    ]
    train_file_paths.sort()

    times = []
    for file_path in train_file_paths:
        time = datetime.strptime(
            os.path.basename(file_path).split("R")[0], "wsa_%Y%m%d%H%M"
        )
        times.append(time)

    return np.array(times), np.array(train_file_paths)


def process_item(i, j, file_path, lats, lons, radius, level):
    br = compute_br_grid(file_path, lats=lats, lons=lons, radius=radius)
    return i, j, contour_lat_extent(br, lats=lats, lons=lons, level=level)


def compute_br_grid(file_path, lats, lons, radius):
    with fits.open(file_path) as fits_file:
        sph_data = fits_file[3].data.copy()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    coeffs_array = np.array([G, H])
    coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array,
        normalization="schmidt",
        r0=1,
    )

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    result = coeffs.expand(
        r=np.full(lat_grid.size, radius, dtype=np.float64),
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        degrees=True,
    )
    br = result.T[0]

    return br.reshape(lat_grid.shape)


def contour_lat_extent(br, lats, lons, level):
    """Return (lat_min, lat_max) over all contour lines of `br` at `level`.

    Returns (nan, nan) if the field has no crossing of `level`.
    """
    generator = contourpy.contour_generator(x=lons, y=lats, z=br)
    lines = generator.lines(level)

    if len(lines) == 0:
        return np.nan, np.nan

    contour_lats = np.concatenate([line[:, 1] for line in lines])
    return contour_lats.min(), contour_lats.max()


if __name__ == "__main__":
    main()
