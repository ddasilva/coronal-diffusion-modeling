import h5py
import json
import numpy as np
import torch
from tqdm import tqdm

from coronal_diffusion.constants import device
from coronal_diffusion.models import DiffusionModel
from coronal_diffusion.utils import GH_to_flat

import config
from config import spharm_fit_mat_path, nlat, nlon, fit_nmax, radii


def get_unscaled_abs():
    # Apply Weighting
    with open(config.scalers_path) as fh:
        scalers_dict = json.load(fh)

    return np.array(scalers_dict["unscaled_abs"])


def main():
    model = DiffusionModel(nmax=fit_nmax)
    coeffs = torch.zeros((fit_nmax + 1, fit_nmax + 1), dtype=torch.complex64)
    unscaled_abs = get_unscaled_abs()

    nrad = radii.size
    ncol = GH_to_flat(coeffs.real, coeffs.imag).size - 1
    nrow = nlat * nlon * nrad
    A = np.zeros((nrow, ncol), dtype=np.float32)
    print("Matrix size:", (nrow, ncol))

    pbar = tqdm(total=ncol, desc="Filling matrix")
    col_counter = 0

    for c in ["g", "h"]:
        for l in range(fit_nmax + 1):
            for m in range(fit_nmax + 1):
                if ((l == 0 and m == 0) and c == "g") or (
                    (l == 0 or m == 0) and c == "h"
                ):
                    # skip G[0, 0,] and H[:1, :1]
                    continue

                if m > l:
                    # skip zeros
                    continue

                coeffs[:] = 0

                if c == "g":
                    coeffs[l, m] = 1
                elif c == "h":
                    coeffs[l, m] = 1j
                else:
                    raise RuntimeError()

                for i in range(nrad):
                    img = make_img(model, coeffs, radii[i]).cpu().numpy()

                    start_row = i * nlat * nlon
                    end_row = start_row + nlat * nlon

                    A[start_row:end_row, col_counter] = img.flatten()

                col_counter += 1
                pbar.update(1)

    assert col_counter == ncol

    # Save to HDF file
    hdf = h5py.File(spharm_fit_mat_path, "w")
    hdf["A"] = A
    hdf.close()

    print(f"Wrote to {spharm_fit_mat_path}")


@torch.no_grad()
def make_img(model, coeffs, r):
    radial_scaling = torch.zeros(coeffs.shape, dtype=torch.float32)

    for n in range(coeffs.shape[-1]):
        radial_scaling[n, :] = 1 / r ** (n + 1)

    img = model.isht(radial_scaling * coeffs).float()

    return img


if __name__ == "__main__":
    main()
