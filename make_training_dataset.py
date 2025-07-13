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

DELTA_ROT = 5


def main(root_dir, out_file):
    # Load the radio flux data
    df_radio = pd.read_csv(
        'data/penticton_radio_flux.csv',
        names=['times', 'observed_flux', 'adjusted_flux'],
        skiprows=1,
        parse_dates=['times']
    )
    df_radio.sort_values(by='times', inplace=True)
    df_radio.drop_duplicates(subset=['times'], keep='first', inplace=True)

    # data is 3X/day
    df_radio['adjusted_flux_smoothed'] = df_radio['adjusted_flux'].rolling(window=3*365).median()

    # Load magnetic field data
    files = glob.glob(f"{root_dir}/*R000*.fits")
    files.sort()

    items = np.zeros((len(files)* 360 // DELTA_ROT, 8281))
    radio_fluxes = np.zeros((len(files)* 360 // DELTA_ROT,))

    counter = 0

    for file in tqdm.tqdm(files, desc="Processing files"):
        time = datetime.datetime.strptime(
            os.path.basename(file).split('R')[0], 'wsa_%Y%m%d%H%M'
        )
        radio_flux = float(np.interp(
            date2num(pd.Timestamp(time)),
            date2num(df_radio['times']),
            df_radio['adjusted_flux_smoothed'],
        ))

        for G, H in enumerate_variations(file):
            items[counter] = (
                np.concatenate(
                    [G[np.tril_indices(G.shape[0])], H[1:,1:][np.tril_indices(H.shape[0]-1)]]
                ).flatten()
            )
            radio_fluxes[counter] = radio_flux
            counter += 1

    # Normalize the radio fluxes
    radio_fluxes = (radio_fluxes - np.min(radio_fluxes)) 
    radio_fluxes /= np.max(radio_fluxes)

    print(items)
    print(radio_fluxes)

    # Write to HDF file
    with h5py.File(out_file, "w") as hdf:
        hdf["X"] = items
        hdf['radio_fluxes'] = radio_fluxes


def enumerate_variations(file_path):
    fits_file = fits.open(file_path)
    sph_data = fits_file[3].data.copy()
    fits_file.close()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    coeffs_array = np.array([G, H])
    coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array, normalization="schmidt", r0=1
    )
    coeffs = coeffs.convert(normalization='ortho')

    yield coeffs.coeffs[0], coeffs.coeffs[1]

    for rot in range(DELTA_ROT, 360, DELTA_ROT):
        rot_coeffs = coeffs.copy().rotate(alpha=rot, beta=0, gamma=0, degrees=True)

        yield rot_coeffs.coeffs[0], rot_coeffs.coeffs[1]


if __name__ == "__main__":
    root_dir = "/home/ubuntu/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
    out_file = "training_dataset.h5"
    main(root_dir, out_file)

    root_dir = "/home/ubuntu/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"
    out_file = "test_dataset.h5"
    main(root_dir, out_file)
