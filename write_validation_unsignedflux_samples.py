import argparse
from pathlib import Path

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
import pyshtools

from coronal_diffusion import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
    )
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument(
        "--match-mode",
        type=str,
        default="exact",
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
        default="data/validation/unsignedflux_samples.h5",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lats, lons, solid_angle = build_flux_grid(args.lat_step, args.lon_step)
    rs = np.array(args.radii, dtype=np.float64)

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
            solid_angle=solid_angle,
        )
        for i, (time, sample_file, sample_time) in enumerate(
            zip(date_range, matched_files, matched_times)
        )
        if sample_file is not None
    ]

    with joblib_progress("Computing unsigned flux samples...", total=len(tasks)):
        results = parallel(tasks)

    valid_results = [result for result in results if result is not None]
    if not valid_results:
        raise ValueError(
            "No harmonic sample files were found for the selected date range."
        )

    nsamples = valid_results[0][4].shape[0]

    flux = np.full((len(date_range), nsamples, rs.size), np.nan, dtype=np.float64)
    contexts = np.full((len(date_range), 6), np.nan, dtype=np.float32)

    n_written = 0
    for index, _, _, context, flux_i in valid_results:
        if flux_i.shape[0] != nsamples:
            raise ValueError("Inconsistent sample count across harmonic sample files.")
        flux[index] = flux_i
        if context is not None:
            contexts[index] = context.astype(np.float32)
        n_written += 1

    with h5py.File(out_file, "w") as hdf:
        hdf["times_d2n"] = date2num(date_range)
        hdf["contexts"] = contexts
        hdf["rs"] = rs
        hdf["unsigned_flux"] = flux
        hdf.attrs["source_samples_dir"] = str(samples_dir)
        hdf.attrs["n_requested_times"] = len(date_range)
        hdf.attrs["n_available_times"] = n_written
        hdf.attrs["match_mode"] = args.match_mode
        hdf.attrs["lat_step"] = args.lat_step
        hdf.attrs["lon_step"] = args.lon_step
        hdf.attrs["flux_definition"] = (
            "Unsigned radial flux, integral of |Br| r^2 dOmega over the sphere "
            "of each radius r (in solar radii), in units of [Br] * Rs^2. "
            "Axes: (time, sample, radius)."
        )

    print(f"Wrote {n_written}/{len(date_range)} timestamps to {out_file}")


def build_flux_grid(lat_step, lon_step):
    """Return cell-centered (lats, lons) in degrees and per-cell solid angles (sr).

    Cell-centered so the surface integral can use exact cell areas.
    """
    lat_edges = np.arange(-90.0, 90.0 + 0.5 * lat_step, lat_step)
    lon_edges = np.arange(-180.0, 180.0 + 0.5 * lon_step, lon_step)
    lats = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lons = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Solid angle of each latitude band cell (per longitude cell), in steradians.
    dlon = np.deg2rad(np.diff(lon_edges))
    band_area = np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    solid_angle = band_area[:, None] * dlon[None, :]

    return lats, lons, solid_angle


def integrate_unsigned_flux(br, solid_angle, rs):
    """Integrate |Br| r^2 dOmega for Br shaped (lat, lon, radius)."""
    abs_br_area = np.abs(br) * solid_angle[:, :, None]
    return abs_br_area.sum(axis=(0, 1)) * rs**2


def process_time_step(
    index, requested_time, sample_file, sample_time, lats, lons, rs, solid_angle
):
    with h5py.File(sample_file, "r") as hdf:
        real = np.array(hdf["real"])
        imag = np.array(hdf["imag"])
        context = np.array(hdf["context"]) if "context" in hdf else None

    real, imag = ensure_sample_axis(real, imag, sample_file)

    flux = np.empty((real.shape[0], rs.size), dtype=np.float64)
    for i in range(real.shape[0]):
        flux[i] = compute_unsigned_flux(
            real=real[i],
            imag=imag[i],
            lats=lats,
            lons=lons,
            rs=rs,
            solid_angle=solid_angle,
        )

    return index, requested_time, sample_time, context, flux


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


def compute_unsigned_flux(real, imag, lats, lons, rs, solid_angle):
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
    br = result.T[0].reshape(lat_grid.shape) 

    return integrate_unsigned_flux(br, solid_angle, rs)


if __name__ == "__main__":
    main()
