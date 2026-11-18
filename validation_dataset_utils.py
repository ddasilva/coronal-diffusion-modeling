from datetime import datetime, timedelta
import os
import re

import h5py
import numpy as np

from coronal_diffusion import utils


def _parse_time_and_realization(file_name):
    stem = os.path.splitext(os.path.basename(file_name))[0]
    match = re.match(r"^(wsa_\d{12})R(\d+)(?:_.+)?$", stem)
    if match is None:
        raise ValueError(
            f"Could not parse time/realization from file name: {file_name}"
        )

    time = datetime.strptime(match.group(1), "wsa_%Y%m%d%H%M")
    realization_index = int(match.group(2))
    return time, realization_index


def collect_dataset_item_metadata(test_wsa_dir):
    file_paths = os.listdir(test_wsa_dir)
    file_paths = [f for f in file_paths if f.endswith(".fits")]
    file_paths.sort()

    if len(file_paths) == 0:
        raise ValueError(f"No FITS test files found in {test_wsa_dir}.")

    items = []
    for file_index, file_name in enumerate(file_paths):
        time, realization_index = _parse_time_and_realization(file_name)

        # prepare_dataset.py writes two rows per file:
        #   even index: original orientation
        #   odd index : flipped orientation
        dataset_index = 2 * file_index

        items.append(
            {
                "time": time,
                "realization_index": realization_index,
                "source_file": os.path.join(test_wsa_dir, file_name),
                "dataset_index": dataset_index,
            }
        )

    return items


def sample_dataset_item_metadata(test_dataset_path, test_wsa_dir, sample_size, seed):
    all_items = collect_dataset_item_metadata(test_wsa_dir)

    _validate_dataset_rows(test_dataset_path, all_items)

    n_available = len(all_items)
    n_selected = min(sample_size, n_available)
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_available, size=n_selected, replace=False)

    return [all_items[int(i)] for i in selected_indices]


def select_random_realization_per_time(
    test_dataset_path,
    test_wsa_dir,
    seed,
    min_realization=0,
    max_realization=11,
):
    all_items = collect_dataset_item_metadata(test_wsa_dir)

    _validate_dataset_rows(test_dataset_path, all_items)

    by_time = {}
    for item in all_items:
        by_time.setdefault(item["time"], []).append(item)

    rng = np.random.default_rng(seed)
    selected_items = []

    for time in sorted(by_time):
        valid_items = [
            item
            for item in by_time[time]
            if min_realization <= item["realization_index"] <= max_realization
        ]
        if len(valid_items) == 0:
            raise ValueError(
                "No realizations in requested range "
                f"[{min_realization}, {max_realization}] for time {time.isoformat()}."
            )

        selected_items.append(valid_items[int(rng.integers(len(valid_items)))])

    return selected_items


def select_random_realization_per_time_augmented(
    test_dataset_path,
    test_wsa_dir,
    seed,
    min_realization=0,
    max_realization=11,
):
    selected_items = select_random_realization_per_time(
        test_dataset_path=test_dataset_path,
        test_wsa_dir=test_wsa_dir,
        seed=seed,
        min_realization=min_realization,
        max_realization=max_realization,
    )

    augmented_items = []
    for item in selected_items:
        for augmentation_index in (0, 1):
            augmented_item = dict(item)
            augmented_item["augmentation_index"] = augmentation_index
            augmented_item["augmentation_name"] = (
                "regular" if augmentation_index == 0 else "flipped"
            )
            augmented_item["dataset_index"] = item["dataset_index"] + augmentation_index
            augmented_items.append(augmented_item)

    rng = np.random.default_rng(seed)
    polarity_flip_flags = rng.random(len(augmented_items)) < 0.5
    for i, apply_flip in enumerate(polarity_flip_flags):
        augmented_items[i]["apply_polarity_inversion"] = bool(apply_flip)

    return augmented_items


def select_random_realization_every_n_days(
    test_dataset_path,
    test_wsa_dir,
    seed,
    step_days=7,
    min_realization=0,
    max_realization=11,
):
    if step_days < 1:
        raise ValueError("step_days must be >= 1.")

    all_items = collect_dataset_item_metadata(test_wsa_dir)

    _validate_dataset_rows(test_dataset_path, all_items)

    by_time = {}
    for item in all_items:
        by_time.setdefault(item["time"], []).append(item)

    candidate_times = sorted(by_time)
    selected_times = []
    for time in candidate_times:
        if not selected_times or (time - selected_times[-1]) >= timedelta(days=step_days):
            selected_times.append(time)

    rng = np.random.default_rng(seed)
    selected_items = []

    for time in selected_times:
        valid_items = [
            item
            for item in by_time[time]
            if min_realization <= item["realization_index"] <= max_realization
        ]
        if len(valid_items) == 0:
            raise ValueError(
                "No realizations in requested range "
                f"[{min_realization}, {max_realization}] for time {time.isoformat()}."
            )

        selected_items.append(valid_items[int(rng.integers(len(valid_items)))])

    return selected_items


def select_random_realization_every_n_days_augmented(
    test_dataset_path,
    test_wsa_dir,
    seed,
    step_days=7,
    min_realization=0,
    max_realization=11,
):
    selected_items = select_random_realization_every_n_days(
        test_dataset_path=test_dataset_path,
        test_wsa_dir=test_wsa_dir,
        seed=seed,
        step_days=step_days,
        min_realization=min_realization,
        max_realization=max_realization,
    )

    augmented_items = []
    for item in selected_items:
        for augmentation_index in (0, 1):
            augmented_item = dict(item)
            augmented_item["augmentation_index"] = augmentation_index
            augmented_item["augmentation_name"] = (
                "regular" if augmentation_index == 0 else "flipped"
            )
            augmented_item["dataset_index"] = item["dataset_index"] + augmentation_index
            augmented_items.append(augmented_item)

    rng = np.random.default_rng(seed)
    polarity_flip_flags = rng.random(len(augmented_items)) < 0.5
    for i, apply_flip in enumerate(polarity_flip_flags):
        augmented_items[i]["apply_polarity_inversion"] = bool(apply_flip)

    return augmented_items


def _validate_dataset_rows(test_dataset_path, all_items):
    with h5py.File(test_dataset_path, "r") as hdf:
        if "X" not in hdf:
            raise ValueError(f"Dataset {test_dataset_path} has no 'X' array.")

        n_rows = hdf["X"].shape[0]
        required_rows = 2 * len(all_items)
        if n_rows < required_rows:
            raise ValueError(
                f"Dataset row count ({n_rows}) is smaller than expected ({required_rows}) "
                "for the FITS files in test_wsa_dir."
            )


def load_coeffs_from_dataset(test_dataset_path, dataset_index):
    with h5py.File(test_dataset_path, "r") as hdf:
        flat = np.array(hdf["X"][dataset_index], dtype=np.float64)

    return utils.unpack_coeffs(flat)
