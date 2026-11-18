import argparse
from datetime import datetime
import os
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import contourpy
import h5py
import numpy as np
import pandas as pd
import pyshtools

from coronal_diffusion import utils
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--match-mode",
        type=str,
        default="nearest",
        choices=["nearest", "exact"],
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="data/validation/harmonic_samples",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/validation/csextent_samples.h5",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1.")

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    test_times = sample_test_times(
        get_test_times(args.test_wsa_dir), args.sample_size, args.seed
    )
    available_times, available_files = load_available_sample_files(samples_dir)
    if len(available_times) == 0:
        raise ValueError(f"No harmonic sample files found in {samples_dir}.")

    matched_files, matched_times = match_sample_files(
        requested_times=test_times,
        available_times=available_times,
        available_files=available_files,
        mode=args.match_mode,
    )

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(process_time_step)(
            index=i,
            requested_time=time,
            sample_file=sample_file,
            sample_time=sample_time,
            lats=lats,
            lons=lons,
            radius=args.radius,
            level=args.level,
        )
        for i, (time, sample_file, sample_time) in enumerate(
            zip(test_times, matched_files, matched_times)
        )
        if sample_file is not None
    ]

    with joblib_progress("Computing current sheet extents...", total=len(tasks)):
        results = parallel(tasks)

    valid_results = [result for result in results if result is not None]
    if not valid_results:
        raise ValueError(
            "No harmonic sample files were found for the selected date range."
        )

    nsamples = valid_results[0][3].shape[0]

    lat_min = np.full((len(test_times), nsamples), np.nan, dtype=np.float32)
    lat_max = np.full((len(test_times), nsamples), np.nan, dtype=np.float32)
    contexts = np.full((len(test_times), 6), np.nan, dtype=np.float32)

    n_written = 0
    for index, _, _, extent_i, context in valid_results:
        if extent_i.shape[0] != nsamples:
            raise ValueError("Inconsistent sample count across harmonic sample files.")
        lat_min[index] = extent_i[:, 0]
        lat_max[index] = extent_i[:, 1]
        if context is not None:
            contexts[index] = context.astype(np.float32)
        n_written += 1

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(test_times)
        hdf["contexts"] = contexts
        hdf["lat_min"] = lat_min
        hdf["lat_max"] = lat_max
        hdf["lat_extent"] = lat_max - lat_min
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["radius"] = np.array([args.radius], dtype=np.float32)
        hdf.attrs["source_samples_dir"] = str(samples_dir)
        hdf.attrs["n_requested_times"] = len(test_times)
        hdf.attrs["n_available_times"] = n_written
        hdf.attrs["sample_size"] = len(test_times)
        hdf.attrs["sample_seed"] = args.seed
        hdf.attrs["match_mode"] = args.match_mode
        hdf.attrs["contour_level"] = args.level
        hdf.attrs["extent_definition"] = (
            "Minimum and maximum latitude (degrees) reached by the Br contour at "
            "the given level and radius. NaN where no contour exists."
        )

    print(f"Wrote {n_written}/{len(test_times)} timestamps to {out_file}")


def get_test_times(test_wsa_dir):
    test_file_paths = os.listdir(test_wsa_dir)
    test_file_paths = [f for f in test_file_paths if f.endswith(".fits") and "R000" in f]
    test_file_paths.sort()

    if len(test_file_paths) == 0:
        raise ValueError(f"No R000 test files found in {test_wsa_dir}.")

    times = []
    for file_path in test_file_paths:
        time = datetime.strptime(
            os.path.basename(file_path).split("R")[0], "wsa_%Y%m%d%H%M"
        )
        times.append(time)

    return np.array(times)


def sample_test_times(test_times, sample_size, seed):
    n_available = len(test_times)
    n_selected = min(sample_size, n_available)
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_available, size=n_selected, replace=False)
    return np.array(test_times)[selected_indices]


def process_time_step(
    index, requested_time, sample_file, sample_time, lats, lons, radius, level
):
    with h5py.File(sample_file, "r") as hdf:
        real = np.array(hdf["real"])
        imag = np.array(hdf["imag"])
        context = np.array(hdf["context"]) if "context" in hdf else None

    real, imag = ensure_sample_axis(real, imag, sample_file)

    # Per sample: (lat_min, lat_max) of the contour.
    extents = np.full((real.shape[0], 2), np.nan, dtype=np.float32)

    for i in range(real.shape[0]):
        br = compute_br_grid(real[i], imag[i], lats=lats, lons=lons, radius=radius)
        extents[i] = contour_lat_extent(br, lats=lats, lons=lons, level=level)

    return index, requested_time, sample_time, extents, context


def load_available_sample_files(samples_dir):
    sample_files = sorted(samples_dir.glob("*.hdf5"))

    times = []
    files = []
    for sample_file in sample_files:
        try:
            timestamp = pd.Timestamp(sample_file.stem)
        except ValueError:
            continue
        times.append(timestamp)
        files.append(sample_file)

    return np.array(times, dtype="datetime64[ns]"), np.array(files, dtype=object)


def match_sample_files(requested_times, available_times, available_files, mode):
    requested_np = np.array(requested_times, dtype="datetime64[ns]")
    requested_ns = requested_np.astype("datetime64[ns]").astype(np.int64)
    available_ns = available_times.astype(np.int64)

    available_by_ns = {
        int(ts_ns): file_path for ts_ns, file_path in zip(available_ns, available_files)
    }
    available_time_by_ns = {
        int(ts_ns): pd.Timestamp(ts_ns, unit="ns") for ts_ns in available_ns
    }

    matched_files = []
    matched_times = []

    for ts_ns in requested_ns:
        ts_key = int(ts_ns)

        if mode == "exact":
            matched_file = available_by_ns.get(ts_key)
            matched_time = available_time_by_ns.get(ts_key)
        else:
            idx = int(np.argmin(np.abs(available_ns - ts_ns)))
            matched_file = available_files[idx]
            matched_time = pd.Timestamp(available_times[idx])

        matched_files.append(matched_file)
        matched_times.append(matched_time)

    return matched_files, matched_times


def ensure_sample_axis(real, imag, sample_file):
    if real.ndim == 2:
        real = real[None, ...]
        imag = imag[None, ...]

    if real.ndim != 3 or imag.ndim != 3:
        raise ValueError(
            f"Expected real/imag to be 2D or 3D arrays in {sample_file}, "
            f"got shapes {real.shape} and {imag.shape}"
        )

    return real, imag


def compute_br_grid(real, imag, lats, lons, radius):
    coeffs = pyshtools.SHMagCoeffs.from_array(
        utils.complex_to_real(real, imag),
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
