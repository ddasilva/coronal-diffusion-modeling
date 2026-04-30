import argparse

from matplotlib.dates import date2num
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyshtools
import h5py
import torch

from coronal_diffusion import constants, models, sampler
import config
from prepare_dataset import load_df_hemi_ss, load_df_hemi_lat


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/experiment54-hemi_9.pth",
    )
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="3MS")
    parser.add_argument('--nsamples', type=int, default=12)
    args = parser.parse_args()

    # Load Hemispheric SS and Latitudinal data
    df_hemi_ss = load_df_hemi_ss()
    df_hemi_lat = load_df_hemi_lat()

    # Prepare to make samples
    sampling_data = sampler.load_sampling_data()
    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    contexts = get_contexts(date_range, df_hemi_ss, df_hemi_lat)
    lats, lons, rs, Bcube = get_field_strength_samples(date_range, contexts, sampling_data, args.weights, args.nsamples)

    # Save the Bcube to an HDF5 file
    out_file = "data/validate_bcube_samples_1e3_wider.h5"

    hdf = h5py.File(out_file, "w")
    hdf['times_d2n'] = date2num(date_range)  # Save times as date numbers
    hdf['contexts'] = contexts
    hdf['lats'] = lats
    hdf['lons'] = lons
    hdf['rs'] = rs
    hdf['Bcube'] = Bcube
    hdf.close()

    print('Wrote to Path:', out_file)


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

    contexts = np.array(contexts)

    return contexts


def get_field_strength_samples(date_range, contexts, sampling_data, weights_file, nsamples):
    lats = np.arange(-89, 90,)
    lons = np.arange(-180, 180)
    rs = np.array([1.025])
    Lats, Lons, Rs = np.meshgrid(lats, lons, rs, indexing="ij")
    Bcube = np.nan * np.zeros((len(date_range), nsamples) + Lats.shape + (3,), dtype=np.float32)

    # Load model
    model = models.DiffusionModel().to(constants.device)
    model.load_state_dict(torch.load(weights_file, map_location=constants.device))
    model.eval()

    # Collect iterable elements 
    iter_elems = []

    for i, (time, context) in enumerate(zip(date_range, contexts)):
        for j in range(nsamples):
            iter_elems.append((i, j, time, context))

    # Loop over iterable elements with progress bar
    for i, j, time, context in tqdm(iter_elems):
        _, (G, H) = sampler.sample(
            model=model,
            sampling_data=sampling_data,
            context=context,
            method="ddpm",
            verbose=False,
        )

        coeffs_array = np.array([G, H])
        coeffs = pyshtools.SHMagCoeffs.from_array(
            coeffs_array,
            normalization="ortho",
            r0=1,
        )

        result = coeffs.expand(
            r=Rs.flatten(), lat=Lats.flatten(), lon=Lons.flatten(), degrees=True
        )

        Br, Btheta, Bphi = result.T
        Bcube[i, j, :, :, :, 0] = Br.reshape(Lats.shape)
        Bcube[i, j, :, :, :, 1] = Btheta.reshape(Lats.shape)
        Bcube[i, j, :, :, :, 2] = Bphi.reshape(Lats.shape)

    return lats, lons, rs, Bcube


if __name__ == "__main__":
    main()
