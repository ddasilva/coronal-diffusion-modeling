import random

import numpy as np
import torch
from pyshtools.backends.shtools import SHctor
import pyshtools

from coronal_diffusion import constants
from coronal_diffusion.constants import device
import config
from config import nmax


def complex_to_real(real, imag):
    """Converts real/imag coefficients (torch_harmonics) to real coefficients
    (shtools)    
    """
    lmax = real.shape[0] - 1

    # 1. Reconstruct the full complex SHCoeffs array
    # PySHTools complex format: index 0 is m >= 0, index 1 is m < 0
    complex_array = np.zeros((2, lmax + 1, lmax + 1), dtype=np.complex128)
    complex_array[0] = real + 1j * imag
    
    for m in range(lmax + 1):
        complex_array[1, :, m] = ((-1)**m) * np.conj(complex_array[0, :, m])
        
    # 2. Create the complex SHCoeffs object (matching prepare_dataset.py csphase=1)
    coeffs_complex = pyshtools.SHCoeffs.from_array(
        complex_array,
        normalization="ortho",
        csphase=1
    )
    
    # 3. Convert back to real coefficients
    coeffs_real = coeffs_complex.convert(kind="real", csphase=1)

    return coeffs_real.coeffs



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


def set_seeds(seed):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
