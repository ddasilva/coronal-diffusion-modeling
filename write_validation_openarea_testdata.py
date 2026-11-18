import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
import numpy as np
from tqdm import tqdm

from coronal_diffusion import visualization_tools as vt
import config
from validation_dataset_utils import (
    load_coeffs_from_dataset,
    select_random_realization_per_time_augmented,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-density", type=int, default=90)
    parser.add_argument("--trace-radius", type=float, default=1.25)
    parser.add_argument("--rk45-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument("--test-dataset-path", type=str, default=config.test_dataset_path)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/open_area_testdata",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_items = select_random_realization_per_time_augmented(
        test_dataset_path=args.test_dataset_path,
        test_wsa_dir=args.test_wsa_dir,
        seed=args.seed,
        min_realization=0,
        max_realization=11,
    )

    np.random.shuffle(sampled_items)
    sampled_items = sampled_items[:args.sample_size]

    for item in tqdm(
        sampled_items,
        total=len(sampled_items),
        desc="Writing open-area testdata",
    ):
        time = item["time"]
        realization_index = item["realization_index"]
        augmentation_index = item["augmentation_index"]
        augmentation_name = item["augmentation_name"]
        source_file = item["source_file"]
        real, imag = load_coeffs_from_dataset(
            args.test_dataset_path, item["dataset_index"]
        )
        if item["apply_polarity_inversion"]:
            real = -real
            imag = -imag

        vis = vt.VisualizerRK45(real, imag, n_jobs=args.rk45_n_jobs)
        lat_values, _, ch_map = vis.get_coronal_holes(
            args.grid_density, r=args.trace_radius
        )
        open_fraction = compute_open_area_fraction(lat_values, ch_map)

        timestamp = time.isoformat()
        file_path = out_dir / f"{timestamp}_R{realization_index:03d}_A{augmentation_index}.hdf5"

        with h5py.File(file_path, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = timestamp
            hdf["realization_index"] = np.int16(realization_index)
            hdf["augmentation_index"] = np.int8(augmentation_index)
            hdf["polarity_inversion_applied"] = np.uint8(
                item["apply_polarity_inversion"]
            )
            hdf["open_fraction"] = open_fraction
            hdf.attrs["grid_density"] = args.grid_density
            hdf.attrs["trace_radius"] = args.trace_radius
            hdf.attrs["source_realization_file"] = str(source_file)
            hdf.attrs["augmentation_name"] = augmentation_name

    print(f"Wrote {len(sampled_items)} files to {out_dir}")


def compute_open_area_fraction(lat_values, ch_map):
    # Area element on a lat/lon grid is proportional to cos(latitude).
    lat_weights = np.cos(np.deg2rad(lat_values))[:, None]
    weights_2d = np.broadcast_to(lat_weights, ch_map.shape)
    return float(np.sum(ch_map * weights_2d) / np.sum(weights_2d))


if __name__ == "__main__":
    main()
