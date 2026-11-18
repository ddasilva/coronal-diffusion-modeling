import argparse
from datetime import datetime
import os
from pathlib import Path

from matplotlib.dates import date2num
import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import pyshtools

from coronal_diffusion import utils
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--lat-step", type=float, default=0.5)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="data/validation/harmonic_samples",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/peaklat_samples",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1.")

    # Fine latitude grid for finding peaks
    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    # Longitude grid
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    test_times = sample_test_times(
        get_test_times(args.test_wsa_dir), args.sample_size, args.seed
    )

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(write_peaklat_file)(
            time=time,
            samples_dir=samples_dir,
            lats=lats,
            lons=lons,
            radius=args.radius,
            out_dir=out_dir,
        )
        for time in test_times
    ]

    with joblib_progress("Writing peak latitude samples...", total=len(tasks)):
        results = parallel(tasks)

    n_written = int(np.sum(results))
    print(f"Wrote {n_written} files to {out_dir}")


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


def write_peaklat_file(time, samples_dir, lats, lons, radius, out_dir):
    timestamp = time.isoformat()
    sample_file = samples_dir / f"{timestamp}.hdf5"

    if not sample_file.exists():
        return 0

    with h5py.File(sample_file, "r") as hdf:
        real = np.array(hdf["real"])
        imag = np.array(hdf["imag"])

    real, imag = ensure_sample_axis(real, imag, sample_file)

    peak_lat_north = np.empty(real.shape[0], dtype=np.float32)
    peak_lat_south = np.empty(real.shape[0], dtype=np.float32)

    for i in range(real.shape[0]):
        peak_lat_north[i], peak_lat_south[i] = compute_peak_lat_grid(
            real[i], imag[i], lats=lats, lons=lons, radius=radius
        )

    out_file = out_dir / f"{timestamp}.hdf5"
    with h5py.File(out_file, "w") as hdf:
        hdf["time_d2n"] = date2num(time)
        hdf["time_str"] = timestamp
        hdf["lats"] = lats
        hdf["radius"] = np.array([radius], dtype=np.float32)
        hdf["peak_lat_north"] = peak_lat_north
        hdf["peak_lat_south"] = peak_lat_south
        hdf.attrs["source_samples_file"] = str(sample_file)
        hdf.attrs["peak_lat_north_definition"] = (
            "Latitude (degrees) where the magnetic field strength is maximum in the "
            "northern hemisphere (0° to 90°), maximizing across all longitudes, "
            "at the specified radius."
        )
        hdf.attrs["peak_lat_south_definition"] = (
            "Latitude (degrees) where the magnetic field strength is maximum in the "
            "southern hemisphere (-90° to 0°), maximizing across all longitudes, "
            "at the specified radius."
        )

    return 1


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


def compute_peak_lat_grid(real, imag, lats, lons, radius):
    """
    Compute hemisphere peak latitudes at the specified radius by maximizing
    magnetic field strength across all longitudes.

    Parameters
    ----------
    real : ndarray
        Real part of spherical harmonic coefficients
    imag : ndarray
        Imaginary part of spherical harmonic coefficients
    lats : ndarray
        Latitude grid (degrees)
    lons : ndarray
        Longitude grid (degrees)
    radius : float
        Radius at which to evaluate field (solar radii)

    Returns
    -------
    peak_lat_north : float
        Peak latitude in northern hemisphere across all longitudes
    peak_lat_south : float
        Peak latitude in southern hemisphere across all longitudes
    """
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

    br, btheta, bphi = result.T
    field_strength = np.sqrt(br**2 + btheta**2 + bphi**2)
    field_strength_grid = field_strength.reshape(lat_grid.shape)

    # Split into northern and southern hemispheres
    # Northern: lats >= 0
    # Southern: lats < 0
    lat_north_mask = lats >= 0
    lat_south_mask = lats < 0

    if lat_north_mask.sum() > 0:
        north_strength = field_strength_grid[lat_north_mask, :]
        north_flat_idx = np.argmax(north_strength)
        north_lat_idx, _ = np.unravel_index(north_flat_idx, north_strength.shape)
        peak_lat_north = np.float32(lats[lat_north_mask][north_lat_idx])
    else:
        peak_lat_north = np.float32(np.nan)

    if lat_south_mask.sum() > 0:
        south_strength = field_strength_grid[lat_south_mask, :]
        south_flat_idx = np.argmax(south_strength)
        south_lat_idx, _ = np.unravel_index(south_flat_idx, south_strength.shape)
        peak_lat_south = np.float32(lats[lat_south_mask][south_lat_idx])
    else:
        peak_lat_south = np.float32(np.nan)

    return peak_lat_north, peak_lat_south


if __name__ == "__main__":
    main()
