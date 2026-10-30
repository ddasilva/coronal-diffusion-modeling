import argparse
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd

from write_validation_unsignedflux_samples import (
    build_flux_grid,
    compute_unsigned_flux,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
    )
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
        default="data/validation/unsignedflux_truncated.h5",
    )
    args = parser.parse_args()

    truncated_dir = Path(args.truncated_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats, lons, solid_angle = build_flux_grid(args.lat_step, args.lon_step)
    rs = np.array(args.radii, dtype=np.float64)

    if lats.size == 0 or lons.size == 0 or rs.size == 0:
        raise ValueError("Latitude, longitude, and radius grids must be non-empty.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(process_time_step)(
            index=i,
            time=time,
            truncated_dir=truncated_dir,
            lats=lats,
            lons=lons,
            rs=rs,
            solid_angle=solid_angle,
        )
        for i, time in enumerate(date_range)
    ]

    with joblib_progress("Computing truncated unsigned flux...", total=len(tasks)):
        results = parallel(tasks)

    valid_results = [result for result in results if result is not None]
    if not valid_results:
        raise ValueError(f"No truncated files found in {truncated_dir}.")

    # Layout matches write_validation_unsignedflux_testdata.py:
    # (n_times, n_realizations, n_radii), with a single realization here.
    flux = np.full((len(date_range), 1, rs.size), np.nan, dtype=np.float64)
    matched_times = np.full(len(date_range), "", dtype=object)
2
    for index, flux_i, matched_wsa_time in valid_results:
        flux[index, 0] = flux_i
        matched_times[index] = matched_wsa_time

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(date_range)
        hdf["rs"] = rs
        hdf["unsigned_flux"] = flux
        hdf["matched_wsa_time_iso"] = np.array(matched_times, dtype="S")
        hdf.attrs["source_truncated_dir"] = str(truncated_dir)
        hdf.attrs["n_requested_times"] = len(date_range)
        hdf.attrs["n_available_times"] = len(valid_results)
        hdf.attrs["lat_step"] = args.lat_step
        hdf.attrs["lon_step"] = args.lon_step
        hdf.attrs["flux_definition"] = (
            "Unsigned radial flux, integral of |Br| r^2 dOmega over the sphere "
            "of each radius r (in solar radii), in units of [Br] * Rs^2. "
            "NaN where no truncated file was found. "
            "Axes: (time, realization, radius)."
        )

    print(f"Wrote {len(valid_results)}/{len(date_range)} timestamps to {out_file}")


def process_time_step(index, time, truncated_dir, lats, lons, rs, solid_angle):
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

    flux = compute_unsigned_flux(
        real=real,
        imag=imag,
        lats=lats,
        lons=lons,
        rs=rs,
        solid_angle=solid_angle,
    )

    return index, flux, matched_wsa_time


if __name__ == "__main__":
    main()
