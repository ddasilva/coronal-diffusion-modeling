import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import pyshtools
import config
from validation_dataset_utils import (
    load_coeffs_from_dataset,
    select_random_realization_per_time_augmented,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-dataset-path", type=str, default=config.test_dataset_path)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/angle_testdata",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    sampled_items.sort(
        key=lambda item: (
            item["time"],
            item["realization_index"],
            item["augmentation_index"],
        )
    )

    parallel = Parallel(n_jobs=args.n_jobs)
    tasks = [
        delayed(write_angle_file)(
            time=item["time"],
            realization_index=item["realization_index"],
            augmentation_index=item["augmentation_index"],
            augmentation_name=item["augmentation_name"],
            source_file=item["source_file"],
            dataset_index=item["dataset_index"],
            test_dataset_path=args.test_dataset_path,
            apply_polarity_inversion=item["apply_polarity_inversion"],
            lats=lats,
            lons=lons,
            radius=args.radius,
            out_dir=out_dir,
        )
        for item in sampled_items
    ]

    with joblib_progress("Writing angle testdata...", total=len(tasks)):
        parallel(tasks)

    print(
        "Wrote "
        f"{len(sampled_items)} files "
        "using all available times, one random realization in [0, 11] per time, "
        "both regular/flipped augmentations, and 50% random polarity inversion "
        f"to {out_dir}"
    )


def write_angle_file(
    time,
    realization_index,
    augmentation_index,
    augmentation_name,
    source_file,
    dataset_index,
    test_dataset_path,
    apply_polarity_inversion,
    lats,
    lons,
    radius,
    out_dir,
):
    real, imag = load_coeffs_from_dataset(test_dataset_path, dataset_index)
    if apply_polarity_inversion:
        real = -real
        imag = -imag
    angles_deg = compute_angle_grid(real, imag, lats=lats, lons=lons, radius=radius)

    timestamp = time.isoformat()
    file_path = out_dir / f"{timestamp}_R{realization_index:03d}_A{augmentation_index}.hdf5"

    with h5py.File(file_path, "w") as hdf:
        hdf["time_d2n"] = date2num(time)
        hdf["time_str"] = timestamp
        hdf["realization_index"] = np.int16(realization_index)
        hdf["augmentation_index"] = np.int8(augmentation_index)
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["radius"] = np.array([radius], dtype=np.float32)
        hdf["angles_deg"] = angles_deg
        hdf["polarity_inversion_applied"] = np.uint8(apply_polarity_inversion)
        hdf["sign_flip_applied"] = np.uint8(apply_polarity_inversion)
        hdf.attrs["source_realization_file"] = str(source_file)
        hdf.attrs["augmentation_name"] = augmentation_name
        hdf.attrs["angle_definition"] = (
            "Angle in degrees between the magnetic-field vector and the outward "
            "radial unit vector."
        )


def compute_angle_grid(real, imag, lats, lons, radius):
    coeffs = pyshtools.SHMagCoeffs.from_array(
        np.array([real, imag]),
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

    br, btheta, bphi = result.T
    field_strength = np.sqrt(br**2 + btheta**2 + bphi**2)
    cos_angle = np.full(br.shape, np.nan, dtype=np.float64)
    nonzero = field_strength > 0
    cos_angle[nonzero] = np.clip(br[nonzero] / field_strength[nonzero], -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(cos_angle))

    return angles_deg.reshape(lat_grid.shape).astype(np.float32)


if __name__ == "__main__":
    main()
