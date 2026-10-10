import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from coronal_diffusion import visualization_tools as vt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--grid-density", type=int, default=90)
    parser.add_argument("--trace-radius", type=float, default=1.25)
    parser.add_argument("--rk45-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="data/validation/harmonic_samples",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/open_area_samples",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    n_written = 0
    for time in tqdm(date_range, desc="Writing open-area samples"):
        timestamp = time.isoformat()
        sample_file = samples_dir / f"{timestamp}.hdf5"

        if not sample_file.exists():
            continue

        with h5py.File(sample_file, "r") as hdf:
            real = np.array(hdf["real"])
            imag = np.array(hdf["imag"])

        if real.ndim == 2:
            real = real[None, ...]
            imag = imag[None, ...]

        if real.ndim != 3 or imag.ndim != 3:
            raise ValueError(
                f"Expected real/imag to be 2D or 3D arrays in {sample_file}, "
                f"got shapes {real.shape} and {imag.shape}"
            )

        nsamples = real.shape[0]
        open_fraction_by_sample = np.zeros(nsamples, dtype=np.float64)

        for i in range(nsamples):
            vis = vt.VisualizerRK45(real[i], imag[i], n_jobs=args.rk45_n_jobs)
            lat_values, _, ch_map = vis.get_coronal_holes(
                args.grid_density, r=args.trace_radius
            )
            open_fraction_by_sample[i] = compute_open_area_fraction(lat_values, ch_map)

        out_file = out_dir / f"{timestamp}.hdf5"
        with h5py.File(out_file, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = timestamp
            hdf["open_fraction"] = np.mean(open_fraction_by_sample)
            hdf["open_fraction_by_sample"] = open_fraction_by_sample
            hdf.attrs["grid_density"] = args.grid_density
            hdf.attrs["trace_radius"] = args.trace_radius
            hdf.attrs["source_samples_file"] = str(sample_file)

        n_written += 1

    print(f"Wrote {n_written} files to {out_dir}")


def compute_open_area_fraction(lat_values, ch_map):
    # Area element on a lat/lon grid is proportional to cos(latitude).
    lat_weights = np.cos(np.deg2rad(lat_values))[:, None]
    weights_2d = np.broadcast_to(lat_weights, ch_map.shape)
    return float(np.sum(ch_map * weights_2d) / np.sum(weights_2d))


if __name__ == "__main__":
    main()
