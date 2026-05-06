import numpy as np
import torch

from coronal_diffusion import constants
from coronal_diffusion.constants import device
import config
from config import nmax


def make_img(model, coeffs, r):

    radial_scaling = torch.zeros(coeffs.shape, dtype=torch.float32, device=device)

    for n in range(coeffs.shape[-1]):
        radial_scaling[n, :] = 1 / r ** (n + 1)

    img = model.isht(radial_scaling * coeffs).float()
    # img = torch.asinh(img) / scalers_std

    return img


def pack_coeffs(real, imag):
    return np.concatenate(
        [
            real[np.tril_indices(real.shape[0])],
            imag[1:, 1:][np.tril_indices(real.shape[0] - 1)],
        ]
    ).flatten()


def unpack_coeffs(flat, nmax=config.nmax):
    real = np.zeros((nmax + 1, nmax + 1))
    imag = np.zeros((nmax + 1, nmax + 1))

    imagtemp = np.zeros((nmax, nmax))
    cutoff = np.tril_indices(nmax + 1)[0].size
    real[np.tril_indices(nmax + 1)] = flat[:cutoff]
    imagtemp[np.tril_indices(nmax)] = flat[cutoff:]
    imag[1:, 1:] = imagtemp

    return real, imag
