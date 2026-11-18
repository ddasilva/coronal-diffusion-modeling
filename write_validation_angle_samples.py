import argparse
from datetime import datetime
import os
from pathlib import Path

from matplotlib.dates import date2num
import h5py
import numpy as np
from tqdm import tqdm
import pyshtools

from coronal_diffusion import utils
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--lat-step", type=float, default=1.0)
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
        default="data/validation/angle_samples",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1.")

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    test_times = sample_test_times(
        get_test_times(args.test_wsa_dir), args.sample_size, args.seed
    )

    n_written = 0
    for time in tqdm(test_times, desc="Writing angle samples"):
        timestamp = time.isoformat()
        sample_file = samples_dir / f"{timestamp}.hdf5"

        if not sample_file.exists():
            continue

        with h5py.File(sample_file, "r") as hdf:
            real = np.array(hdf["real"])
            imag = np.array(hdf["imag"])

        real, imag = ensure_sample_axis(real, imag, sample_file)

        angles_deg = np.empty((real.shape[0], lats.size, lons.size), dtype=np.float32)

        for i in range(real.shape[0]):
            angles_deg[i] = compute_angle_grid(
                real[i], imag[i], lats=lats, lons=lons, radius=args.radius
            )

        out_file = out_dir / f"{timestamp}.hdf5"
        with h5py.File(out_file, "w") as hdf:
            hdf["time_d2n"] = date2num(time)
            hdf["time_str"] = timestamp
            hdf["lats"] = lats
            hdf["lons"] = lons
            hdf["radius"] = np.array([args.radius], dtype=np.float32)
            hdf["angles_deg"] = angles_deg
            hdf.attrs["source_samples_file"] = str(sample_file)
            hdf.attrs["angle_definition"] = (
                "Angle in degrees between the magnetic-field vector and the outward "
                "radial unit vector."
            )

        n_written += 1

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


def compute_angle_grid(real, imag, lats, lons, radius):
    real_coeffs = utils.complex_to_real(real, imag)
    coeffs = pyshtools.SHMagCoeffs.from_array(
        real_coeffs,
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
