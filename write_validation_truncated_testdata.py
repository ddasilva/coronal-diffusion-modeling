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
import torch
from tqdm import tqdm

from coronal_diffusion import constants, sampler
from coronal_diffusion.models import DiffusionModel
import config
import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--realization", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=str, default="data/validation/truncated_testdata"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    val_times, val_file_paths = get_r000_val_file_paths()

    scalers_dict = train.get_scalers(config)
    sampling_data = sampler.load_sampling_data()
    model = DiffusionModel().to(constants.device)
    model.eval()

    val_times_d2n = date2num(val_times)

    for time in tqdm(date_range, desc="Writing truncated harmonic test set"):
        idx = np.argmin(np.abs(val_times_d2n - date2num(time)))
        base_file = os.path.join(config.val_wsa_dir, val_file_paths[idx])
        file_path = base_file.replace("R000", f"R{args.realization:03d}")

        original_real, original_imag = read_harmonics_from_fits(file_path)
        truncated_real, truncated_imag = get_truncated_fit(
            original_real,
            original_imag,
            model,
            scalers_dict,
            sampling_data,
        )

        timestamp = time.isoformat()
        out_file = out_dir / f"{timestamp}.hdf5"

        with h5py.File(out_file, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = time.isoformat()
            hdf["original_real"] = original_real
            hdf["original_imag"] = original_imag
            hdf["truncated_real"] = truncated_real
            hdf["truncated_imag"] = truncated_imag
            hdf.attrs["source_fits"] = file_path
            hdf.attrs["target_time_iso"] = time.isoformat()
            hdf.attrs["matched_wsa_time_iso"] = val_times[idx].isoformat()
            hdf.attrs["realization"] = args.realization

    print(f"Wrote {len(date_range)} files to {out_dir}")


def get_r000_val_file_paths():
    val_file_paths = os.listdir(config.val_wsa_dir)
    val_file_paths = [f for f in val_file_paths if f.endswith(".fits") and "R000" in f]
    val_file_paths.sort()

    times = []
    for file_path in val_file_paths:
        time = datetime.strptime(
            os.path.basename(file_path).split("R")[0], "wsa_%Y%m%d%H%M"
        )
        times.append(time)

    return np.array(times), np.array(val_file_paths)


def read_harmonics_from_fits(file_path):
    with fits.open(file_path) as fits_file:
        sph_data = fits_file[3].data.copy()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    coeffs_array = np.array([G, H])
    coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array, normalization="schmidt", r0=1
    )
    coeffs = coeffs.convert(normalization="ortho")
    G, H = coeffs.coeffs

    return G, H


def get_truncated_fit(real, imag, model, scalers_dict, sampling_data):
    coeffs = torch.from_numpy(real + imag * 1j).unsqueeze(0).to(constants.device)

    potential_images = (
        train.get_potential_images(model, coeffs, config.radii, scalers_dict)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    truncated_real, truncated_imag = sampler.spherical_harm_fit_smaller(
        potential_images, sampling_data, verbose=False
    )

    return truncated_real, truncated_imag


if __name__ == "__main__":
    main()
