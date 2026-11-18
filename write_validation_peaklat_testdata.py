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
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--lat-step", type=float, default=0.5)
    parser.add_argument("--lon-step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-dataset-path", type=str, default=config.test_dataset_path)
    parser.add_argument("--test-wsa-dir", type=str, default=config.test_wsa_dir)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/validation/peaklat_testdata",
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

    parallel = Parallel(n_jobs=args.n_jobs)
    tasks = [
        delayed(write_peaklat_file)(
            time=item["time"],
            realization_index=item["realization_index"],
            augmentation_index=item["augmentation_index"],
            augmentation_name=item["augmentation_name"],
            source_file=item["source_file"],
            dataset_index=item["dataset_index"],
            apply_polarity_inversion=item["apply_polarity_inversion"],
            test_dataset_path=args.test_dataset_path,
            lats=lats,
            lons=lons,
            radius=args.radius,
            out_dir=out_dir,
        )
        for item in sampled_items
    ]

    with joblib_progress("Writing peak latitude testdata...", total=len(tasks)):
        parallel(tasks)

    print(f"Wrote {len(sampled_items)} files to {out_dir}")


def write_peaklat_file(
    time,
    realization_index,
    augmentation_index,
    augmentation_name,
    source_file,
    dataset_index,
    apply_polarity_inversion,
    test_dataset_path,
    lats,
    lons,
    radius,
    out_dir,
):
    real, imag = load_coeffs_from_dataset(test_dataset_path, dataset_index)
    if apply_polarity_inversion:
        real = -real
        imag = -imag
    peak_lat_north, peak_lat_south = compute_peak_lat_grid(
        real, imag, lats=lats, lons=lons, radius=radius
    )

    timestamp = time.isoformat()
    file_path = out_dir / f"{timestamp}_R{realization_index:03d}_A{augmentation_index}.hdf5"

    with h5py.File(file_path, "w") as hdf:
        hdf["time_d2n"] = date2num(time)
        hdf["time_str"] = timestamp
        hdf["realization_index"] = np.int16(realization_index)
        hdf["augmentation_index"] = np.int8(augmentation_index)
        hdf["polarity_inversion_applied"] = np.uint8(apply_polarity_inversion)
        hdf["lats"] = lats
        hdf["radius"] = np.array([radius], dtype=np.float32)
        hdf["peak_lat_north"] = peak_lat_north
        hdf["peak_lat_south"] = peak_lat_south
        hdf.attrs["source_realization_file"] = str(source_file)
        hdf.attrs["augmentation_name"] = augmentation_name
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
