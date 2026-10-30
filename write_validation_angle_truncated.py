import argparse
from pathlib import Path

from matplotlib.dates import date2num
import h5py
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
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
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--lat-step", type=float, default=1.0)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument(
        "--truncated-dir",
        type=str,
        default="data/validation/truncated_testdata",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/angle_truncated",
    )
    args = parser.parse_args()

    truncated_dir = Path(args.truncated_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lats = np.arange(-89.0, 90.0, args.lat_step, dtype=np.float64)
    lons = np.arange(-180.0, 180.0, args.lon_step, dtype=np.float64)

    if lats.size == 0 or lons.size == 0:
        raise ValueError("Latitude and longitude grids must contain at least one point.")

    date_range = pd.date_range(args.start_time, args.end_time, freq=args.freq)

    parallel = Parallel(n_jobs=args.n_jobs, prefer="processes")
    tasks = [
        delayed(write_angle_file)(
            time=time,
            truncated_dir=truncated_dir,
            lats=lats,
            lons=lons,
            radius=args.radius,
            out_dir=out_dir,
        )
        for time in date_range
    ]

    with joblib_progress("Writing angle truncated...", total=len(tasks)):
        results = parallel(tasks)

    n_written = int(np.sum(results))
    print(f"Wrote {n_written} files to {out_dir}")


def write_angle_file(time, truncated_dir, lats, lons, radius, out_dir):
    timestamp = time.isoformat()
    truncated_file = truncated_dir / f"{timestamp}.hdf5"

    if not truncated_file.exists():
        return 0

    # Only the truncated (fit) harmonics are used. The original_* datasets are in a
    # different coefficient layout and must not be mixed with these.
    with h5py.File(truncated_file, "r") as hdf:
        real = np.array(hdf["truncated_real"])
        imag = np.array(hdf["truncated_imag"])
        matched_wsa_time = hdf.attrs.get("matched_wsa_time_iso", "")

    if real.ndim != 2 or imag.ndim != 2:
        raise ValueError(
            f"Expected truncated_real/imag to be 2D arrays in {truncated_file}, "
            f"got shapes {real.shape} and {imag.shape}"
        )

    angles_deg = compute_angle_grid(real, imag, lats=lats, lons=lons, radius=radius)

    out_file = out_dir / f"{timestamp}.hdf5"
    with h5py.File(out_file, "w") as hdf:
        hdf["time_d2n"] = date2num(time)
        hdf["time_str"] = timestamp
        hdf["lats"] = lats
        hdf["lons"] = lons
        hdf["radius"] = np.array([radius], dtype=np.float32)
        hdf["angles_deg"] = angles_deg
        hdf.attrs["source_truncated_file"] = str(truncated_file)
        hdf.attrs["matched_wsa_time_iso"] = matched_wsa_time
        hdf.attrs["angle_definition"] = (
            "Angle in degrees between the magnetic-field vector and the outward "
            "radial unit vector."
        )

    return 1


def compute_angle_grid(real, imag, lats, lons, radius):
    # Imported here because SHctor is Fortran-backed and cannot be pickled by joblib.
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
    cos_angle = np.full(br.shape, np.nan, dtype=np.float64)
    nonzero = field_strength > 0
    cos_angle[nonzero] = np.clip(br[nonzero] / field_strength[nonzero], -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(cos_angle))

    return angles_deg.reshape(lat_grid.shape).astype(np.float32)


if __name__ == "__main__":
    main()
