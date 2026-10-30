import argparse
from datetime import datetime
import os
from pathlib import Path

from astropy.io import fits
from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
import pyshtools
from tqdm import tqdm

from coronal_diffusion import constants
from coronal_diffusion import visualization_tools as vt
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--grid-density", type=int, default=90)
    parser.add_argument("--trace-radius", type=float, default=1.25)
    parser.add_argument("--rk45-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/open_area_testdata",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    train_times, train_file_paths = get_r000_train_file_paths()
    train_times_d2n = date2num(train_times)

    for time in tqdm(date_range, desc="Writing open-area testdata"):
        idx = np.argmin(np.abs(train_times_d2n - date2num(time)))            
        base_file = os.path.join(config.train_wsa_dir, train_file_paths[idx])

        open_fraction_by_realization = np.zeros(constants.N_REAL, dtype=np.float64)

        #for j in range(constants.N_REAL):
        for j in [0]:
            real_file_path = base_file.replace("R000", f"R{j:03d}")
            real, imag = read_ortho_harmonics(real_file_path)

            vis = vt.VisualizerRK45(real, imag, n_jobs=args.rk45_n_jobs)
            lat_values, lon_values, ch_map = vis.get_coronal_holes(
                args.grid_density, r=args.trace_radius
            )
            open_fraction_by_realization[j] = compute_open_area_fraction(lat_values, ch_map)

        timestamp = time.isoformat()
        file_path = out_dir / f"{timestamp}.hdf5"

        with h5py.File(file_path, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = timestamp
            hdf["open_fraction"] = np.mean(open_fraction_by_realization)
            hdf["open_fraction_by_realization"] = open_fraction_by_realization
            hdf.attrs["grid_density"] = args.grid_density
            hdf.attrs["trace_radius"] = args.trace_radius
            hdf.attrs["matched_wsa_time_iso"] = train_times[idx].isoformat()

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

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    coeffs = pyshtools.SHMagCoeffs.from_array(
        np.array([G, H]),
        normalization="schmidt",
        r0=1,
    )
    coeffs = coeffs.convert(normalization="ortho")

    real, imag = coeffs.coeffs
    return real, imag


def compute_open_area_fraction(lat_values, ch_map):
    # Area element on a lat/lon grid is proportional to cos(latitude).
    lat_weights = np.cos(np.deg2rad(lat_values))[:, None]
    weights_2d = np.broadcast_to(lat_weights, ch_map.shape)
    return float(np.sum(ch_map * weights_2d) / np.sum(weights_2d))


if __name__ == "__main__":
    main()
