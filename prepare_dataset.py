import glob
import os
import datetime

from astropy.io import fits
import numpy as np
import pyshtools
import tqdm
import h5py
import pandas as pd
from matplotlib.dates import date2num
from scipy.interpolate import make_smoothing_spline

from coronal_diffusion.constants import N_CONTEXT
import config


def main(root_dir, out_file):
    # Issue warning for the adventourous traveler
    print(
        "Warning: using hardcoded trailing polarity, valid for training data between "
        "2010-2020 only."
    )

    # Load hemispheric median latitudes of sunspots
    df_hemi_lat = load_df_hemi_lat()

    # Load hemispheric sunspot numbers
    df_hemi_ss = load_df_hemi_ss()

    # Load magnetic field data
    files = glob.glob(f"{root_dir}/*.fits")
    files.sort()

    hdf = h5py.File(out_file, "w")
    items_shape = (2 * len(files), config.X_SIZE)
    items = hdf.create_dataset("X", items_shape, dtype=np.float32)
    context = np.zeros((2 * len(files), N_CONTEXT), dtype=np.float32)

    counter = 0

    for file in tqdm.tqdm(files):
        results = process_file(file, df_hemi_ss, df_hemi_lat)

        for real, imag, cur_context in results:
            items[counter] = np.concatenate(
                [
                    real[np.tril_indices(config.nmax + 1)],
                    imag[1:, 1:][np.tril_indices(config.nmax)],
                ]
            ).flatten()
            context[counter] = cur_context

            counter += 1

    assert counter == items_shape[0]

    hdf["context"] = context

    hdf.close()


def enumerate_variations(
    file_path,
    hemi_ss_n,
    hemi_ss_s,
    hemi_med_lat_n,
    hemi_med_lat_s,
    hemi_trail_pol_n,
    hemi_trail_pol_s,
):
    fits_file = fits.open(file_path)
    sph_data = fits_file[3].data.copy()
    fits_file.close()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    coeffs_array = np.array([G, H])
    coeffs = pyshtools.SHCoeffs.from_array(
        coeffs_array,
        normalization="schmidt",
        csphase=-1,
    )
    coeffs = coeffs.convert(normalization="ortho", kind="complex", csphase=-1)

    # Normal orientation
    context = [
        hemi_ss_n,
        hemi_ss_s,
        hemi_med_lat_n,
        hemi_med_lat_s,
        hemi_trail_pol_n,
        hemi_trail_pol_s,
    ]
    yield coeffs.coeffs[0].real, coeffs.coeffs[0].imag, context

    # Top/Down flip
    flip_coeffs = coeffs.copy().rotate(alpha=0, beta=180, gamma=0, degrees=True)
    context = [
        hemi_ss_s,
        hemi_ss_n,
        hemi_med_lat_s,
        hemi_med_lat_n,
        hemi_trail_pol_s,
        hemi_trail_pol_n,
    ]
    yield flip_coeffs.coeffs[0].real, flip_coeffs.coeffs[0].imag, context


def process_file(file, df_hemi_ss, df_hemi_lat):
    # Get time of the file
    time = datetime.datetime.strptime(
        os.path.basename(file).split("R")[0], "wsa_%Y%m%d%H%M"
    )

    # Hemispheric sunspot number
    i = df_hemi_ss.times.searchsorted(time)
    if i in (0, len(df_hemi_ss)):
        raise RuntimeError(f"Insufficient hemispheric sunspot number data: {time}")

    hemi_ss_n = df_hemi_ss.iloc[i]["sunspot_north_smooth"]
    hemi_ss_s = df_hemi_ss.iloc[i]["sunspot_south_smooth"]

    # Hemispheric sunspot median latitude
    i = df_hemi_lat.times.searchsorted(time)
    if i in (0, len(df_hemi_lat)):
        raise RuntimeError(
            f"Insufficient median latitude hemispheric sunspot data {time}"
        )

    hemi_med_lat_n = abs(df_hemi_lat.iloc[i]["MedianLatNorth_Smoothed"])
    hemi_med_lat_s = abs(df_hemi_lat.iloc[i]["MedianLatSouth_Smoothed"])

    # Hemispheric trailing polarity (hardcoded for now)
    # these will also be flipped in the dataloader if harmonics are inverted from RNG
    assert 2010 <= time.year <= 2020
    hemi_trail_pol_n = 1
    hemi_trail_pol_s = -1

    # Collect results
    results = []

    for real, imag, cur_context in enumerate_variations(
        file,
        hemi_ss_n,
        hemi_ss_s,
        hemi_med_lat_n,
        hemi_med_lat_s,
        hemi_trail_pol_n,
        hemi_trail_pol_s,
    ):
        results.append((real, imag, cur_context))

    return results


def load_df_hemi_ss():
    column_names = [
        "year",
        "month",
        "day",
        "date_fraction",
        "sunspot_total",
        "sunspot_north",
        "sunspot_south",
        "std_total",
        "std_north",
        "std_south",
        "n_obs_total",
        "n_obs_north",
        "n_obs_south",
        "provisional_marker",
    ]

    df_hemi_ss = pd.read_csv(
        "data/SN_d_hem_V2.0.txt",
        sep=r"\s+",  # Split on whitespace
        names=column_names,
        na_values=-1,  # Convert -1 to NaN (optional)
    )
    df_hemi_ss["times"] = pd.to_datetime(df_hemi_ss[["year", "month", "day"]])
    df_hemi_ss = df_hemi_ss[["times", "sunspot_north", "sunspot_south"]]

    # Smoth hemispheric sunspot number
    smoothing_lam = 1e6

    sp = make_smoothing_spline(
        date2num(df_hemi_ss.times), df_hemi_ss.sunspot_north, lam=smoothing_lam
    )
    df_hemi_ss["sunspot_north_smooth"] = sp(date2num(df_hemi_ss.times))

    sp = make_smoothing_spline(
        date2num(df_hemi_ss.times), df_hemi_ss.sunspot_south, lam=smoothing_lam
    )
    df_hemi_ss["sunspot_south_smooth"] = sp(date2num(df_hemi_ss.times))

    return df_hemi_ss


def load_df_hemi_lat():
    return pd.read_csv("data/hemispheric_median_lats.csv", parse_dates=["times"])


if __name__ == "__main__":
    main(config.train_wsa_dir, config.train_dataset_path)
    main(config.test_wsa_dir, config.test_dataset_path)
