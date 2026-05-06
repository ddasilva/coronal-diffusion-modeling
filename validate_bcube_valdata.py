import argparse
from datetime import datetime
import os

from astropy.io import fits
import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
import pyshtools
from tqdm import tqdm

from coronal_diffusion import constants
import config


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="3MS")
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()

    # Prepare date range
    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    # Get Bcube for validation data
    val_times, val_file_paths = get_r000_val_file_paths()

    lats, lons, rs, Bcube = get_field_strength_valdata(
        date_range, val_times, val_file_paths, args.n_jobs
    )

    # Save the Bcube to an HDF5 file
    out_file = "data/validate_bcube_valdata.h5"

    hdf = h5py.File(out_file, "w")
    hdf["times_d2n"] = date2num(date_range)  # Save times as date numbers
    hdf["lats"] = lats
    hdf["lons"] = lons
    hdf["rs"] = rs
    hdf["Bcube"] = Bcube
    hdf.close()

    print("Wrote to Path:", out_file)


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

    times = np.array(times)
    val_file_paths = np.array(val_file_paths)

    return times, val_file_paths


def get_field_strength_valdata(date_range, val_times, val_file_paths, n_jobs):
    lats = np.arange(
        -89,
        90,
    )
    lons = np.arange(-180, 180)
    rs = np.array([1.025])
    Lats, Lons, Rs = np.meshgrid(lats, lons, rs, indexing="ij")

    Bcube = np.nan * np.zeros(
        (len(date_range), constants.N_REAL) + Lats.shape + (3,), dtype=np.float32
    )

    # Gather items to iterae
    iter_items = []
    val_times_d2n = date2num(val_times)

    for i, time in enumerate(date_range):
        # Find the closest WSA file
        idx = np.argmin(np.abs(val_times_d2n - date2num(time)))
        file_path = os.path.join(config.val_wsa_dir, val_file_paths[idx])

        for j in range(constants.N_REAL):
            real_file_path = file_path.replace("R000", f"R{j:03d}")
            iter_items.append((i, j, real_file_path))

    # Do processing in parallel
    parallel = Parallel(n_jobs=n_jobs)
    tasks = [
        delayed(parallel_target)(i, j, real_file_path, lats, lons, rs)
        for i, j, real_file_path in iter_items
    ]
    with joblib_progress("Calculating Bcube...", total=len(tasks)):
        results = parallel(tasks)

    for i, j, Br, Btheta, Bphi in results:
        Bcube[i, j, :, :, :, 0] = Br.reshape(Lats.shape)
        Bcube[i, j, :, :, :, 1] = Btheta.reshape(Lats.shape)
        Bcube[i, j, :, :, :, 2] = Bphi.reshape(Lats.shape)

    return lats, lons, rs, Bcube


def parallel_target(i, j, file_path, lats, lons, rs):
    Lats, Lons, Rs = np.meshgrid(lats, lons, rs, indexing="ij")

    fits_file = fits.open(file_path)
    sph_data = fits_file[3].data.copy()
    fits_file.close()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    G = G[: config.fit_nmax + 1, : config.fit_nmax + 1]
    H = H[: config.fit_nmax + 1, : config.fit_nmax + 1]

    coeffs_array = np.array([G, H])
    coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array, normalization="schmidt", r0=1
    )

    result = coeffs.expand(
        r=Rs.flatten(), lat=Lats.flatten(), lon=Lons.flatten(), degrees=True
    )

    Br, Btheta, Bphi = result.T

    return i, j, Br, Btheta, Bphi


if __name__ == "__main__":
    main()
