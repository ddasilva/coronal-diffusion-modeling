import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

from matplotlib.dates import date2num
import numpy as np
from tqdm import tqdm
import h5py
import torch

sys.path.append('..')
from coronal_diffusion import constants, models, sampler, utils
from prepare_dataset import load_df_hemi_ss, load_df_hemi_lat
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/experiment59-cs-fix-and-context-scaling_16.pth",
    )
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/harmonic_samples",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1.")

    utils.set_seeds(0)
    
    # Load context data and build timestamps from available test files.
    df_hemi_ss = load_df_hemi_ss()
    df_hemi_lat = load_df_hemi_lat()
    test_times = sample_test_times(
        get_test_times(args.test_wsa_dir), args.sample_size, args.seed
    )
    contexts = get_contexts(test_times, df_hemi_ss, df_hemi_lat)

    # Load model and sampling assets.
    sampling_data = sampler.load_sampling_data()
    model = models.DiffusionModel().to(constants.device)
    model.load_state_dict(torch.load(args.weights, map_location=constants.device))
    model.eval()

    for time, context in tqdm(
        zip(test_times, contexts), total=len(test_times), desc="Writing harmonic samples"
    ):
        real_samples = []
        imag_samples = []
        resid_samples = []
        
        for _ in range(args.nsamples):
            _, (real, imag), resid = sampler.sample(
                model=model,
                sampling_data=sampling_data,
                context=context,
                method="ddpm",
                verbose=False,
                return_residual=True,
            )
            real_samples.append(real)
            imag_samples.append(imag)
            resid_samples.append(resid)
            
        real_samples = np.stack(real_samples, axis=0)
        imag_samples = np.stack(imag_samples, axis=0)
        resid_samples = np.stack(resid_samples, axis=0)
        
        timestamp = time.isoformat()
        file_path = out_dir / f"{timestamp}.hdf5"

        with h5py.File(file_path, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = time.isoformat()
            hdf["context"] = context
            hdf["real"] = real_samples.squeeze()
            hdf["imag"] = imag_samples.squeeze()
            hdf["resid"] = resid_samples.squeeze()
            
    print(f"Wrote {len(test_times)} files to {out_dir}")


def get_test_times(test_wsa_dir):
    test_file_paths = os.listdir(test_wsa_dir)
    test_file_paths = [f for f in test_file_paths if f.endswith(".fits") and "R000" in f]
    test_file_paths.sort()

    if len(test_file_paths) == 0:
        raise ValueError(f"No R000 test files found in {test_wsa_dir}.")

    times = []
    for file_path in test_file_paths:
        time = datetime.strptime(
            os.path.basename(file_path).split("R")[0], "wsa_%Y%m%d%H%M"
        )
        times.append(time)

    return np.array(times)


def sample_test_times(test_times, sample_size, seed):
    n_available = len(test_times)
    n_selected = min(sample_size, n_available)
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_available, size=n_selected, replace=False)
    return np.array(test_times)[selected_indices]


def get_contexts(times, df_hemi_ss, df_hemi_lat):
    contexts = []

    for time in times:
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
