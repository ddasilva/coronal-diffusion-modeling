import numpy as np
import torch

from coronal_diffusion import constants
from coronal_diffusion.constants import device
import config
from config import nmax


def make_img(coeffs, r):

    radial_scaling = torch.zeros(coeffs.shape, dtype=torch.float32, device=device)
    
    for n in range(coeffs.shape[-1]):
        radial_scaling[n, :] = 1 / r**(n + 1)

    img = model.isht(radial_scaling * coeffs).float()
    #img = torch.asinh(img) / scalers_std

    return img


def GH_to_flat(G, H):
    return np.concatenate(
        [G[np.tril_indices(config.nmax + 1)], H[1:, 1:][np.tril_indices(config.nmax)]]
    ).flatten()


def flat_to_GH(flat):
    G = np.zeros((nmax + 1, nmax + 1))
    H = np.zeros((nmax + 1, nmax + 1))
    Htemp = np.zeros((nmax, nmax))
    cutoff = np.tril_indices(nmax + 1)[0].size
    G[np.tril_indices(nmax + 1)] = flat[:cutoff]
    Htemp[np.tril_indices(nmax)] = flat[cutoff:]
    H[1:, 1:] = Htemp

    return G, H


def flat_to_2dcomplex(flat_vec):
    batch_size = flat_vec.shape[0]
    cutoff = np.tril_indices(nmax + 1)[0].size
    G = torch.zeros((batch_size, nmax + 1, nmax + 1)).to(device)
    H = torch.zeros((batch_size, nmax + 1, nmax + 1)).to(device)

    for i in range(batch_size):
        G[i][np.tril_indices(nmax + 1)] = flat_vec[i, :cutoff]
        H[i][1:, 1:][np.tril_indices(nmax)] = flat_vec[i, cutoff:]

    coeffs = torch.zeros((batch_size, nmax + 1, nmax + 1), dtype=torch.complex64).to(
        device
    )
    coeffs += G
    coeffs += H * 1j

    return coeffs
