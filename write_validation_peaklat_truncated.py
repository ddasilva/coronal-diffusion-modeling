import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import pandas as pd

from write_validation_peaklat_samples import compute_peak_lat_grid


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
        "--truncated-dir",
        type=str,
        default="data/validation/truncated_testdata",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/peaklat_truncated",
    )
    args = parser.parse_args()

    truncated_dir = Path(args.truncated_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fine latitude grid for finding peaks
    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    # Longitude grid
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(write_peaklat_file)(
            time=time,
            truncated_dir=truncated_dir,
            lats=lats,
            lons=lons,
            radius=args.radius,
            out_dir=out_dir,
        )
        for time in date_range
    ]

    with joblib_progress("Writing peak latitude truncated...", total=len(tasks)):
        results = parallel(tasks)

    n_written = int(np.sum(results))
    print(f"Wrote {n_written} files to {out_dir}")


def write_peaklat_file(time, truncated_dir, lats, lons, radius, out_dir):
    timestamp = time.isoformat()
    truncated_file = truncated_dir / f"{timestamp}.hdf5"

    if not truncated_file.exists():
        return 0

    # Only the truncated (fit) harmonics are used. The original_* datasets are in a
    # different coefficient layout and must not be mixed with these.
    with h5py.File(truncated_file, "r") as hdf:
        real = np.array(hdf["truncated_real"])
        imag = np.array(hdf["truncated_imag"])
        matched_wsa_time = hdf.attrs.get("matched_wsa_time_iso", "")

    if real.ndim != 2 or imag.ndim != 2:
        raise ValueError(
            f"Expected truncated_real/imag to be 2D arrays in {truncated_file}, "
            f"got shapes {real.shape} and {imag.shape}"
        )

    peak_lat_north, peak_lat_south = compute_peak_lat_grid(
        real, imag, lats=lats, lons=lons, radius=radius
    )

    out_file = out_dir / f"{timestamp}.hdf5"
    with h5py.File(out_file, "w") as hdf:
        hdf["time_d2n"] = date2num(time)
        hdf["time_str"] = timestamp
        hdf["lats"] = lats
        hdf["radius"] = np.array([radius], dtype=np.float32)
        hdf["peak_lat_north"] = np.array([peak_lat_north], dtype=np.float32)
        hdf["peak_lat_south"] = np.array([peak_lat_south], dtype=np.float32)
        hdf.attrs["source_truncated_file"] = str(truncated_file)
        hdf.attrs["matched_wsa_time_iso"] = matched_wsa_time
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

    return 1


if __name__ == "__main__":
    main()
