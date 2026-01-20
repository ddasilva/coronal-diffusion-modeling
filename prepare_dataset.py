
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

from coronal_diffusion.constants import N_CONTEXT
import config


def main(root_dir, out_file):
    # Load the radio flux data (3X / day)
    df_radio = pd.read_csv(
        "data/penticton_radio_flux.csv",
        names=["times", "observed_flux", "adjusted_flux"],
        skiprows=1,
        parse_dates=["times"],
    )
    df_radio.sort_values(by="times", inplace=True)
    df_radio.drop_duplicates(subset=["times"], keep="first", inplace=True)


    df_radio["adjusted_flux_smoothed"] = (
        df_radio["adjusted_flux"].rolling(window=3 * 365).median()
    )

    df_radio['times_date2num'] = date2num(df_radio['times'])
    
    # Load solar wind observations and positions
    df_sw = pd.read_csv('data/ace_obs.csv.gz')
    df_sw['times'] = pd.to_datetime(df_sw['times'])
    df_sw['times_date2num'] = date2num(df_sw['times'])
    df_sw = df_sw.interpolate() # interp over NaN's
    
    df_locs = pd.read_csv('data/ace_locs.csv')
    df_locs['time'] = pd.to_datetime(df_locs['time'])
    df_locs['time_date2num'] = date2num(df_locs['time'])
    
    # Load magnetic field data
    files = glob.glob(f"{root_dir}/*.fits")
    files.sort()

    hdf = h5py.File(out_file, "w")
    items_shape = (len(files), config.X_SIZE)
    items = hdf.create_dataset("X", items_shape, dtype=np.float32)
    context = np.zeros(
        (len(files), N_CONTEXT), dtype=np.float32
    )

    counter = 0

    for file in tqdm.tqdm(files):
        results = process_file(file, df_radio, df_sw, df_locs)

        for G, H, cur_context in results:
            items[counter] = np.concatenate(
                [
                    G[np.tril_indices(config.nmax + 1)],
                    H[1:, 1:][np.tril_indices(config.nmax)],
                ]
            ).flatten()
            context[counter] = cur_context

            counter += 1

    assert counter == items_shape[0]
    
    # Normalize the radio fluxes and write
    radio_fluxes = context[:, 0]
    #radio_fluxes[radio_fluxes == 0] = radio_fluxes[radio_fluxes > 0].min()

    print(radio_fluxes[radio_fluxes == 0])
    radio_fluxes = radio_fluxes - np.min(radio_fluxes)
    radio_fluxes /= np.max(radio_fluxes)
    context[:, 0] = radio_fluxes
    
    hdf["context"] = context

    hdf.close()


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
    coeffs = coeffs.convert(normalization="ortho")
    #coeffs = coeffs.rotate(alpha=-earth_lon, beta=0, gamma=0, degrees=True)
    yield coeffs.coeffs[0], coeffs.coeffs[1]
    
    #flip_coeffs = coeffs.copy().rotate(alpha=0, beta=180, gamma=0, degrees=True)
    #yield flip_coeffs.coeffs[0], flip_coeffs.coeffs[1]


def process_file(file, df_radio, df_sw, df_locs):
    time = datetime.datetime.strptime(
        os.path.basename(file).split("R")[0], "wsa_%Y%m%d%H%M"
    )
    
    radio_flux = float(
        np.interp(
            date2num(pd.Timestamp(time)),
            df_radio['times_date2num'],
            df_radio["adjusted_flux_smoothed"],
        )
    )

    Vsw = float(
        np.interp(
            date2num(pd.Timestamp(time)),
            df_sw["times_date2num"],
            df_sw['Vp_obs']
        )
    )

    earth_lon_idx = np.argmin(
        df_locs['time'] - time
    )
    earth_lon = df_locs['sat_lon'].iloc[earth_lon_idx]
    
    earth_lat = float(
        np.interp(
            date2num(pd.Timestamp(time)),
            df_locs["time_date2num"],
            df_locs['sat_lat'],
        )
    )
    
    results = []

    cur_context = np.array([
        radio_flux,
        Vsw,
        earth_lat,
        earth_lon,
    ])
    
    for G, H in enumerate_variations(file):
        results.append((G, H, cur_context))

    return results


if __name__ == "__main__":
    main(config.train_wsa_dir, config.train_dataset_path)
    main(config.test_wsa_dir, config.test_dataset_path)
