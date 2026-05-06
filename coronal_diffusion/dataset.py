import glob
import os
import random

from astropy.io import fits
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from coronal_diffusion.utils import unpack_coeffs
from config import nmax


# class CoronalFieldDataset(Dataset):

#     def __init__(self, root_dir):
#         self.files = glob.glob(f"{root_dir}/*R000*.fits")
#         assert len(self) > 0
#         self.cache = {}

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):

#         if idx in self.cache:
#             return self.cache[idx]

#         wsa_path = os.path.join(self.files[idx])
#         fits_file = fits.open(wsa_path)
#         sph_data = fits_file[3].data.copy()
#         fits_file.close()
#         output = torch.from_numpy(
#             np.array(
#                 [
#                     sph_data[0, :, :][np.triu_indices(sph_data.shape[1])],
#                     sph_data[1, :, :][np.triu_indices(sph_data.shape[1])],
#                 ]
#             )
#         ).flatten()

#         self.cache[idx] = output

#         return output


class CoronalFieldDatasetHDF(Dataset):

    def __init__(self, path, augment=True):
        self.path = path
        self.augment = augment
        self.hdf = None
        self.X = None
        self.context = None
        
    def __len__(self):
        # Open quickly just to get length, clean open/close
        with h5py.File(self.path, "r") as f:
            return f["X"].shape[0]

    def _init_hdf5(self):
        # Lazy initialization inside worker processes
        if self.hdf is None:
            self.hdf = h5py.File(self.path, "r")
            self.X = self.hdf["X"]
            self.context = self.hdf["context"]

    def __getitem__(self, idx):
        self._init_hdf5()
        
        coeffs_real, coeffs_imag = unpack_coeffs(self.X[idx])

        coeffs = torch.complex(
            torch.from_numpy(coeffs_real),
            torch.from_numpy(coeffs_imag)
        )
        
        context = torch.from_numpy(np.array([self.context[idx]])).float()

        if self.augment and random.random() > 0.5:
            # Invert field line polarity
            coeffs *= -1

            # Change the hemispheric leading polarities
            context[:, -2] *= -1
            context[:, -1] *= -1
        
        return coeffs, context
