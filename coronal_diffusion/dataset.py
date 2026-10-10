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
