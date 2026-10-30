import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from coronal_diffusion import visualization_tools as vt
from write_validation_openarea_samples import compute_open_area_fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--grid-density", type=int, default=90)
    parser.add_argument("--trace-radius", type=float, default=1.25)
    parser.add_argument("--rk45-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--truncated-dir",
        type=str,
        default="data/validation/truncated_testdata",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/open_area_truncated",
    )
    args = parser.parse_args()

    truncated_dir = Path(args.truncated_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    n_written = 0
    for time in tqdm(date_range, desc="Writing open-area truncated"):
        timestamp = time.isoformat()
        truncated_file = truncated_dir / f"{timestamp}.hdf5"

        if not truncated_file.exists():
            continue

        # Only the truncated (fit) harmonics are used. The original_* datasets are in
        # a different coefficient layout and must not be mixed with these.
        with h5py.File(truncated_file, "r") as hdf:
            real = np.array(hdf["truncated_real"])
            imag = np.array(hdf["truncated_imag"])
            matched_wsa_time = hdf.attrs.get("matched_wsa_time_iso", "")

        if real.ndim != 2 or imag.ndim != 2:
            raise ValueError(
                f"Expected truncated_real/imag to be 2D arrays in {truncated_file}, "
                f"got shapes {real.shape} and {imag.shape}"
            )

        vis = vt.VisualizerRK45(real, imag, n_jobs=args.rk45_n_jobs)
        lat_values, _, ch_map = vis.get_coronal_holes(
            args.grid_density, r=args.trace_radius
        )
        open_fraction = compute_open_area_fraction(lat_values, ch_map)

        out_file = out_dir / f"{timestamp}.hdf5"
        with h5py.File(out_file, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = timestamp
            hdf["open_fraction"] = open_fraction
            hdf.attrs["grid_density"] = args.grid_density
            hdf.attrs["trace_radius"] = args.trace_radius
            hdf.attrs["source_truncated_file"] = str(truncated_file)
            hdf.attrs["matched_wsa_time_iso"] = matched_wsa_time

        n_written += 1

    print(f"Wrote {n_written} files to {out_dir}")


if __name__ == "__main__":
    main()
