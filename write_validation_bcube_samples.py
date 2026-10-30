import argparse
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
import pyshtools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="3MS")
    parser.add_argument("--n-jobs", type=int, default=16)
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
        default="data/validation/bcube_samples_exp59.h5",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats = np.arange(-89, 90)
    lons = np.arange(-180, 180)
    rs = np.array([1.00], dtype=np.float64)

    if lats.size == 0 or lons.size == 0 or rs.size == 0:
        raise ValueError("Latitude, longitude, and radius grids must be non-empty.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)
    available_times, available_files = load_available_sample_files(samples_dir)
    if len(available_times) == 0:
        raise ValueError(f"No harmonic sample files found in {samples_dir}.")

    matched_files, matched_times = match_sample_files(
        requested_times=date_range,
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
            rs=rs,
        )
        for i, (time, sample_file, sample_time) in enumerate(
            zip(date_range, matched_files, matched_times)
        )
        if sample_file is not None
    ]

    with joblib_progress("Computing Bcube samples...", total=len(tasks)):
        results = parallel(tasks)

    valid_results = [result for result in results if result is not None]
    if not valid_results:
        raise ValueError(
            "No harmonic sample files were found for the selected date range."
        )

    first_bcube = valid_results[0][4]
    nsamples = first_bcube.shape[0]
    bcube_shape = first_bcube.shape[1:]

    bcube = np.full(
        (len(date_range), nsamples) + bcube_shape,
        np.nan,
        dtype=np.float32,
    )
    contexts = np.full((len(date_range), 6), np.nan, dtype=np.float32)

    n_written = 0
    for index, _, _, context, bcube_i in valid_results:
        if bcube_i.shape[0] != nsamples or bcube_i.shape[1:] != bcube_shape:
            raise ValueError(
                "Inconsistent sample shape across harmonic sample files."
            )
        bcube[index] = bcube_i
        if context is not None:
            contexts[index] = context.astype(np.float32)
        n_written += 1

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(date_range)
        hdf["contexts"] = contexts
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["rs"] = rs
        hdf["Bcube"] = bcube
        hdf.attrs["source_samples_dir"] = str(samples_dir)
        hdf.attrs["n_requested_times"] = len(date_range)
        hdf.attrs["n_available_times"] = n_written
        hdf.attrs["match_mode"] = args.match_mode

    print(f"Wrote {n_written}/{len(date_range)} timestamps to {out_file}")


def process_time_step(index, requested_time, sample_file, sample_time, lats, lons, rs):

    with h5py.File(sample_file, "r") as hdf:
        real = np.array(hdf["real"])
        imag = np.array(hdf["imag"])
        context = np.array(hdf["context"]) if "context" in hdf else None

    real, imag = ensure_sample_axis(real, imag, sample_file)

    lat_grid, lon_grid, r_grid = np.meshgrid(lats, lons, rs, indexing="ij")
    output_shape = lat_grid.shape

    rs_flat = r_grid.ravel().astype(np.float64, copy=False)
    lats_flat = lat_grid.ravel().astype(np.float64, copy=False)
    lons_flat = lon_grid.ravel().astype(np.float64, copy=False)

    bcube = np.empty((real.shape[0],) + output_shape + (3,), dtype=np.float32)

    for i in range(real.shape[0]):
        br, btheta, bphi = expand_field_components(
            real=real[i],
            imag=imag[i],
            rs_flat=rs_flat,
            lats_flat=lats_flat,
            lons_flat=lons_flat,
            output_shape=output_shape,
        )
        bcube[i, :, :, :, 0] = br
        bcube[i, :, :, :, 1] = btheta
        bcube[i, :, :, :, 2] = bphi

    return index, requested_time, sample_time, context, bcube


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


def expand_field_components(real, imag, rs_flat, lats_flat, lons_flat, output_shape):
    # Imported here (not at module level) so cloudpickle never has to serialize
    # the Fortran-backed function when shipping this function to joblib workers.
    from pyshtools.backends.shtools import SHctor

    # `real`/`imag` are the ortho-normalized complex coefficients stored by
    # prepare_dataset.py. Stack them as a *float* array (not complex) as SHctor
    # expects, then convert to real coefficients. Use the default csphase (+1,
    # no Condon-Shortley phase): that matches the model's transforms
    # (csphase=False) and the physical WSA field, and makes this expansion
    # identical (scale 1) to write_validation_bcube_testdata.py.
    complex_array = np.empty((2,) + real.shape, dtype=np.float64)
    complex_array[0] = real
    complex_array[1] = imag

    real_coeffs = SHctor(complex_array)
    coeffs = pyshtools.SHMagCoeffs.from_array(
        real_coeffs,
        normalization="ortho",
        r0=1,
    )

    # Fortran-backed SHTools objects are created and consumed inside this worker call.
    result = coeffs.expand(
        r=rs_flat,
        lat=lats_flat,
        lon=lons_flat,
        degrees=True,
    )
    br, btheta, bphi = result.T

    return (
        br.reshape(output_shape).astype(np.float32),
        btheta.reshape(output_shape).astype(np.float32),
        bphi.reshape(output_shape).astype(np.float32),
    )


if __name__ == "__main__":
    main()
