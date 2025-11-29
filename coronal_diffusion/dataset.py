import glob
import os
import random

from astropy.io import fits
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from coronal_diffusion.utils import flat_to_GH
from config import nmax


class CoronalFieldDataset(Dataset):

    def __init__(self, root_dir):
        self.files = glob.glob(f"{root_dir}/*R000*.fits")
        assert len(self) > 0
        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if idx in self.cache:
            return self.cache[idx]

        wsa_path = os.path.join(self.files[idx])
        fits_file = fits.open(wsa_path)
        sph_data = fits_file[3].data.copy()
        fits_file.close()
        output = torch.from_numpy(
            np.array(
                [
                    sph_data[0, :, :][np.triu_indices(sph_data.shape[1])],
                    sph_data[1, :, :][np.triu_indices(sph_data.shape[1])],
                ]
            )
        ).flatten()

        self.cache[idx] = output

        return output


class CoronalFieldDatasetHDF(Dataset):

    def __init__(self, path):
        self.hdf = h5py.File(path, "r")
        self.X = self.hdf["X"]
        self.radio_fluxes = self.hdf["radio_fluxes"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        G, H = flat_to_GH(self.X[idx])

        coeffs = torch.zeros((nmax + 1, nmax + 1), dtype=torch.complex64)
        coeffs += G + H * 1j

        radio_flux = torch.from_numpy(np.array([self.radio_fluxes[idx]])).float()

        if random.random() > 0.5:
            coeffs *= -1

        return coeffs, radio_flux
