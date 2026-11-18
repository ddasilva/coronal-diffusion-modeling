import argparse
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import h5py
import numpy as np
import pyshtools

from coronal_diffusion import utils
from write_validation_unsignedflux_samples import (
    build_flux_grid,
    integrate_unsigned_flux,
)
import config
from validation_dataset_utils import (
    load_coeffs_from_dataset,
    select_random_realization_per_time_augmented,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
    )
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--test-dataset-path", type=str, default=config.test_dataset_path)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/validation/unsignedflux_testdata.h5",
    )
    args = parser.parse_args()

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats, lons, solid_angle = build_flux_grid(args.lat_step, args.lon_step)
    rs = np.array(args.radii, dtype=np.float64)

    if lats.size == 0 or lons.size == 0 or rs.size == 0:
        raise ValueError("Latitude, longitude, and radius grids must be non-empty.")

    sampled_items = select_random_realization_per_time_augmented(
        test_dataset_path=args.test_dataset_path,
        test_wsa_dir=args.test_wsa_dir,
        seed=args.seed,
        min_realization=0,
        max_realization=11,
    )
    n_selected = len(sampled_items)

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(process_item)(
            sample_index,
            args.test_dataset_path,
            item["dataset_index"],
            item["apply_polarity_inversion"],
            lats,
            lons,
            rs,
            solid_angle,
        )
        for sample_index, item in enumerate(sampled_items)
    ]
    with joblib_progress("Computing unsigned flux...", total=len(tasks)):
        results = parallel(tasks)

    selected_times = np.array([item["time"] for item in sampled_items])
    selected_realizations = np.array(
        [item["realization_index"] for item in sampled_items], dtype=np.int16
    )
    selected_augmentations = np.array(
        [item["augmentation_index"] for item in sampled_items], dtype=np.int8
    )
    selected_polarity_inversions = np.array(
        [item["apply_polarity_inversion"] for item in sampled_items], dtype=np.uint8
    )

    flux = np.full((n_selected, rs.size), np.nan, dtype=np.float64)
    for sample_index, flux_ij in results:
        flux[sample_index] = flux_ij

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(selected_times)
        hdf["realization_index"] = selected_realizations
        hdf["augmentation_index"] = selected_augmentations
        hdf["polarity_inversion_applied"] = selected_polarity_inversions
        hdf["rs"] = rs
        hdf["unsigned_flux"] = flux
        hdf.attrs["source_test_dataset"] = args.test_dataset_path
        hdf.attrs["sample_size"] = n_selected
        hdf.attrs["sample_seed"] = args.seed
        hdf.attrs["lat_step"] = args.lat_step
        hdf.attrs["lon_step"] = args.lon_step
        hdf.attrs["flux_definition"] = (
            "Unsigned radial flux, integral of |Br| r^2 dOmega over the sphere "
            "of each radius r (in solar radii), in units of [Br] * Rs^2. "
            "Axes: (sample, radius)."
        )

    print(f"Wrote {n_selected} sampled items to {out_file}")


def process_item(
    sample_index,
    test_dataset_path,
    dataset_index,
    apply_polarity_inversion,
    lats,
    lons,
    rs,
    solid_angle,
):
    return sample_index, compute_unsigned_flux(
        test_dataset_path,
        dataset_index,
        apply_polarity_inversion,
        lats,
        lons,
        rs,
        solid_angle,
    )


def compute_unsigned_flux(
    test_dataset_path,
    dataset_index,
    apply_polarity_inversion,
    lats,
    lons,
    rs,
    solid_angle,
):
    real, imag = load_coeffs_from_dataset(test_dataset_path, dataset_index)
    if apply_polarity_inversion:
        real = -real
        imag = -imag
    coeffs = pyshtools.SHMagCoeffs.from_array(
        utils.complex_to_real(real, imag),
        normalization="ortho",
        r0=1,
    )

    lat_grid, lon_grid, r_grid = np.meshgrid(lats, lons, rs, indexing="ij")
    result = coeffs.expand(
        r=r_grid.ravel(),
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        degrees=True,
    )

    # Same rescaling as write_validation_bcube_testdata.py, so the field is on the
    # same scale as the harmonic samples. It matters here because the flux uses |Br|.
    #result /= 4 * np.pi * np.sqrt(2)
    br = result.T[0].reshape(lat_grid.shape)

    return integrate_unsigned_flux(br, solid_angle, rs)


if __name__ == "__main__":
    main()
