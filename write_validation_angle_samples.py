import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyshtools
from pyshtools.backends.shtools import SHctor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, default="2010-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--freq", type=str, default="7D")
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
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

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    n_written = 0
    for time in tqdm(date_range, desc="Writing angle samples"):
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
    complex_array = np.zeros((2,) + real.shape, dtype=complex)
    complex_array[0] = real + 1j * imag

    real_coeffs = SHctor(complex_array)
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