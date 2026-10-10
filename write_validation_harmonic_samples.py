import argparse
from pathlib import Path
import sys

from matplotlib.dates import date2num
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import torch

sys.path.append('..')
from coronal_diffusion import constants, models, sampler
from prepare_dataset import load_df_hemi_ss, load_df_hemi_lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/experiment59-cs-fix-and-context-scaling_16.pth",
    )
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/harmonic_samples",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load context data and build weekly date range.
    df_hemi_ss = load_df_hemi_ss()
    df_hemi_lat = load_df_hemi_lat()
    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    contexts = get_contexts(date_range, df_hemi_ss, df_hemi_lat)

    # Load model and sampling assets.
    sampling_data = sampler.load_sampling_data()
    model = models.DiffusionModel().to(constants.device)
    model.load_state_dict(torch.load(args.weights, map_location=constants.device))
    model.eval()

    for time, context in tqdm(
        zip(date_range, contexts), total=len(date_range), desc="Writing harmonic samples"
    ):
        real_samples = []
        imag_samples = []

        for _ in range(args.nsamples):
            _, (real, imag) = sampler.sample(
                model=model,
                sampling_data=sampling_data,
                context=context,
                method="ddpm",
                verbose=False,
            )
            real_samples.append(real)
            imag_samples.append(imag)

        real_samples = np.stack(real_samples, axis=0)
        imag_samples = np.stack(imag_samples, axis=0)

        timestamp = time.isoformat()
        file_path = out_dir / f"{timestamp}.hdf5"

        with h5py.File(file_path, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = time.isoformat()
            hdf["context"] = context
            hdf["real"] = real_samples.squeeze()
            hdf["imag"] = imag_samples.squeeze()

    print(f"Wrote {len(date_range)} files to {out_dir}")


def get_contexts(date_range, df_hemi_ss, df_hemi_lat):
    contexts = []

    for time in date_range:
        time_d2n = date2num(time)
        ssn_north = np.interp(
            time_d2n, date2num(df_hemi_ss.times), df_hemi_ss["sunspot_north_smooth"]
        )
        ssn_south = np.interp(
            time_d2n, date2num(df_hemi_ss.times), df_hemi_ss["sunspot_south_smooth"]
        )
        hemi_lat_north = np.interp(
            time_d2n,
            date2num(df_hemi_lat.times),
            df_hemi_lat["MedianLatNorth_Smoothed"],
        )
        hemi_lat_south = -np.interp(
            time_d2n,
            date2num(df_hemi_lat.times),
            df_hemi_lat["MedianLatSouth_Smoothed"],
        )
        context = [ssn_north, ssn_south, hemi_lat_north, hemi_lat_south, -1, 1]
        contexts.append(context)

    return np.array(contexts)


if __name__ == "__main__":
    main()
