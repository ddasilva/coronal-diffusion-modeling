import argparse

import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import numpy as np
import pyshtools

from coronal_diffusion import utils
import config
from validation_dataset_utils import (
    load_coeffs_from_dataset,
    select_random_realization_per_time_augmented,
)


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-dataset-path", type=str, default=config.test_dataset_path)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    args = parser.parse_args()

    sampled_items = select_random_realization_per_time_augmented(
        test_dataset_path=args.test_dataset_path,
        test_wsa_dir=args.test_wsa_dir,
        seed=args.seed,
        min_realization=0,
        max_realization=11,
    )

    (
        sampled_times,
        sampled_realizations,
        sampled_augmentations,
        sampled_polarity_inversions,
        lats,
        lons,
        rs,
        Bcube,
    ) = get_field_strength_testdata(
        sampled_items,
        args.test_dataset_path,
        args.n_jobs,
    )

    # Save the Bcube to an HDF5 file
    out_file = "data/validation/bcube_testdata.h5"

    hdf = h5py.File(out_file, "w")
    hdf["times_d2n"] = date2num(sampled_times)
    hdf["realization_index"] = sampled_realizations
    hdf["augmentation_index"] = sampled_augmentations
    hdf["polarity_inversion_applied"] = sampled_polarity_inversions
    hdf["lats"] = lats
    hdf["lons"] = lons
    hdf["rs"] = rs
    hdf["Bcube"] = Bcube
    hdf.attrs["source_test_dataset"] = args.test_dataset_path
    hdf.attrs["sample_size"] = Bcube.shape[0]
    hdf.attrs["sample_seed"] = args.seed
    hdf.attrs["realization_range"] = "0-11"
    hdf.close()

    print("Wrote to Path:", out_file)


def get_field_strength_testdata(sampled_items, test_dataset_path, n_jobs):
    lats = np.arange(
        -89,
        90,
    )
    lons = np.arange(-180, 180)
    rs = np.array([1.0, 1.5, 2.0, 2.5])
    Lats, Lons, Rs = np.meshgrid(lats, lons, rs, indexing="ij")

    sampled_times = np.array([item["time"] for item in sampled_items])
    sampled_realizations = np.array(
        [item["realization_index"] for item in sampled_items], dtype=np.int16
    )
    sampled_augmentations = np.array(
        [item["augmentation_index"] for item in sampled_items], dtype=np.int8
    )
    sampled_polarity_inversions = np.array(
        [item["apply_polarity_inversion"] for item in sampled_items], dtype=np.uint8
    )

    n_selected = len(sampled_items)

    Bcube = np.nan * np.zeros((n_selected,) + Lats.shape + (3,), dtype=np.float32)

    # Do processing in parallel
    parallel = Parallel(n_jobs=n_jobs)
    tasks = [
        delayed(parallel_target)(
            sample_index,
            test_dataset_path,
            item["dataset_index"],
            item["apply_polarity_inversion"],
            lats,
            lons,
            rs,
        )
        for sample_index, item in enumerate(sampled_items)
    ]
    with joblib_progress("Calculating Bcube...", total=len(tasks)):
        results = parallel(tasks)

    for sample_index, Br, Btheta, Bphi in results:
        Bcube[sample_index, :, :, :, 0] = Br.reshape(Lats.shape)
        Bcube[sample_index, :, :, :, 1] = Btheta.reshape(Lats.shape)
        Bcube[sample_index, :, :, :, 2] = Bphi.reshape(Lats.shape)

    return (
        sampled_times,
        sampled_realizations,
        sampled_augmentations,
        sampled_polarity_inversions,
        lats,
        lons,
        rs,
        Bcube,
    )


def parallel_target(
    sample_index,
    test_dataset_path,
    dataset_index,
    apply_polarity_inversion,
    lats,
    lons,
    rs,
):
    Lats, Lons, Rs = np.meshgrid(lats, lons, rs, indexing="ij")

    real, imag = load_coeffs_from_dataset(test_dataset_path, dataset_index)
    if apply_polarity_inversion:
        real = -real
        imag = -imag
    coeffs_array = utils.complex_to_real(real, imag)

    mag_coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array,
        normalization="ortho",
        r0=1,
    )

    result = mag_coeffs.expand(
        r=Rs.flatten(), lat=Lats.flatten(), lon=Lons.flatten(), degrees=True
    )

    #result /= 4 * np.pi * np.sqrt(2)
    
    Br, Btheta, Bphi = result.T
    

    return sample_index, Br, Btheta, Bphi


if __name__ == "__main__":
    main()
