import argparse
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import contourpy
import h5py
import numpy as np
import pandas as pd
import pyshtools

from coronal_diffusion import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument(
        "--truncated-dir",
        type=str,
        default="data/validation/truncated_testdata",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/validation/csextent_truncated.h5",
    )
    args = parser.parse_args()

    truncated_dir = Path(args.truncated_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(process_time_step)(
            index=i,
            time=time,
            truncated_dir=truncated_dir,
            lats=lats,
            lons=lons,
            radius=args.radius,
            level=args.level,
        )
        for i, time in enumerate(date_range)
    ]

    with joblib_progress("Computing truncated current sheet extents...", total=len(tasks)):
        results = parallel(tasks)

    valid_results = [result for result in results if result is not None]
    if not valid_results:
        raise ValueError(f"No truncated files found in {truncated_dir}.")

    # Layout matches write_validation_csextent_testdata.py: (n_times, n_realizations).
    lat_min = np.full((len(date_range), 1), np.nan, dtype=np.float32)
    lat_max = np.full((len(date_range), 1), np.nan, dtype=np.float32)
    matched_times = np.full(len(date_range), "", dtype=object)

    for index, extent, matched_wsa_time in valid_results:
        lat_min[index, 0] = extent[0]
        lat_max[index, 0] = extent[1]
        matched_times[index] = matched_wsa_time

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(date_range)
        hdf["lat_min"] = lat_min
        hdf["lat_max"] = lat_max
        hdf["lat_extent"] = lat_max - lat_min
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["radius"] = np.array([args.radius], dtype=np.float32)
        hdf["matched_wsa_time_iso"] = np.array(matched_times, dtype="S")
        hdf.attrs["source_truncated_dir"] = str(truncated_dir)
        hdf.attrs["n_requested_times"] = len(date_range)
        hdf.attrs["n_available_times"] = len(valid_results)
        hdf.attrs["contour_level"] = args.level
        hdf.attrs["extent_definition"] = (
            "Minimum and maximum latitude (degrees) reached by the Br contour at "
            "the given level and radius. NaN where no contour exists or no "
            "truncated file was found."
        )

    print(f"Wrote {len(valid_results)}/{len(date_range)} timestamps to {out_file}")


def process_time_step(index, time, truncated_dir, lats, lons, radius, level):
    truncated_file = truncated_dir / f"{time.isoformat()}.hdf5"

    if not truncated_file.exists():
        return None

    # Only the truncated (fit) harmonics are used. The original_* datasets are in a
    # different coefficient layout and must not be mixed with these.
    with h5py.File(truncated_file, "r") as hdf:
        real = np.array(hdf["truncated_real"])
        imag = np.array(hdf["truncated_imag"])
        matched_wsa_time = str(hdf.attrs.get("matched_wsa_time_iso", ""))

    if real.ndim != 2 or imag.ndim != 2:
        raise ValueError(
            f"Expected truncated_real/imag to be 2D arrays in {truncated_file}, "
            f"got shapes {real.shape} and {imag.shape}"
        )

    br = compute_br_grid(real, imag, lats=lats, lons=lons, radius=radius)
    extent = contour_lat_extent(br, lats=lats, lons=lons, level=level)

    return index, extent, matched_wsa_time


def compute_br_grid(real, imag, lats, lons, radius):
    coeffs = pyshtools.SHMagCoeffs.from_array(
        utils.complex_to_real(real, imag),
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
