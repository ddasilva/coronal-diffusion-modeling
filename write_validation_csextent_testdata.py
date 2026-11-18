import argparse
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import contourpy
import h5py
import numpy as np
import pyshtools

from coronal_diffusion import utils
import config
from validation_dataset_utils import (
    load_coeffs_from_dataset,
    select_random_realization_per_time_augmented,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--test-dataset-path", type=str, default=config.test_dataset_path)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/validation/csextent_testdata.h5",
    )
    args = parser.parse_args()

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

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
            args.radius,
            args.level,
        )
        for sample_index, item in enumerate(sampled_items)
    ]
    with joblib_progress("Computing current sheet extents...", total=len(tasks)):
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

    lat_min = np.full(n_selected, np.nan, dtype=np.float32)
    lat_max = np.full(n_selected, np.nan, dtype=np.float32)
    for sample_index, extent in results:
        lat_min[sample_index] = extent[0]
        lat_max[sample_index] = extent[1]

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(selected_times)
        hdf["realization_index"] = selected_realizations
        hdf["augmentation_index"] = selected_augmentations
        hdf["polarity_inversion_applied"] = selected_polarity_inversions
        hdf["lat_min"] = lat_min
        hdf["lat_max"] = lat_max
        hdf["lat_extent"] = lat_max - lat_min
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["radius"] = np.array([args.radius], dtype=np.float32)
        hdf.attrs["source_test_dataset"] = args.test_dataset_path
        hdf.attrs["sample_size"] = n_selected
        hdf.attrs["sample_seed"] = args.seed
        hdf.attrs["contour_level"] = args.level
        hdf.attrs["extent_definition"] = (
            "Minimum and maximum latitude (degrees) reached by the Br contour at "
            "the given level and radius. NaN where no contour exists."
        )

    print(f"Wrote {n_selected} sampled items to {out_file}")


def process_item(
    sample_index,
    test_dataset_path,
    dataset_index,
    apply_polarity_inversion,
    lats,
    lons,
    radius,
    level,
):
    br = compute_br_grid(
        test_dataset_path,
        dataset_index,
        apply_polarity_inversion,
        lats=lats,
        lons=lons,
        radius=radius,
    )
    return sample_index, contour_lat_extent(br, lats=lats, lons=lons, level=level)


def compute_br_grid(
    test_dataset_path,
    dataset_index,
    apply_polarity_inversion,
    lats,
    lons,
    radius,
):
    real, imag = load_coeffs_from_dataset(test_dataset_path, dataset_index)
    if apply_polarity_inversion:
        real = -real
        imag = -imag
    coeffs_array = utils.complex_to_real(real, imag)
    coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array,
        normalization="ortho",
        r0=1,
    )

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    result = coeffs.expand(
        r=np.full(lat_grid.size, radius, dtype=np.float64),
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        degrees=True,
    )
    br = result.T[0]

    return br.reshape(lat_grid.shape)


def contour_lat_extent(br, lats, lons, level):
    """Return (lat_min, lat_max) over all contour lines of `br` at `level`.

    Returns (nan, nan) if the field has no crossing of `level`.
    """
    generator = contourpy.contour_generator(x=lons, y=lats, z=br)
    lines = generator.lines(level)

    if len(lines) == 0:
        return np.nan, np.nan

    contour_lats = np.concatenate([line[:, 1] for line in lines])
    return contour_lats.min(), contour_lats.max()


if __name__ == "__main__":
    main()
