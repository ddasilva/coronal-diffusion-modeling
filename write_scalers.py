import h5py
import json
import numpy as np
import dask.array as da
import tqdm
import torch

from coronal_diffusion.dataset import CoronalFieldDatasetHDF
from coronal_diffusion.constants import asinh_sf, device
from coronal_diffusion.models import DiffusionModel
import config


def main():
    hdf = h5py.File(config.test_dataset_path)
    train_dataset = CoronalFieldDatasetHDF(config.train_dataset_path)
    model = DiffusionModel().to(device)
    
    sum_sq = 0
    total = 0
    
    for coeffs, _ in tqdm.tqdm(train_dataset):
        # mean is zero by construction
        coeffs = coeffs.to(device)
        img = model.isht(coeffs).float()
        img = torch.asinh(img / asinh_sf)
        sum_sq += torch.sum(img**2).item()
        total += img.shape[0] * img.shape[1]
    
    std = np.sqrt(sum_sq / (total - 1))
        
    out = {
        # "mean": mean.tolist(),
        "std": std
    }

    with open(config.scalers_path, "w") as fh:
        json.dump(out, fh, indent=4)


if __name__ == "__main__":
    main()
    print(f"Scalers written to {config.scalers_path}")
