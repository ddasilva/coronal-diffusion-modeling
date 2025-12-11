import json
import numpy as np
import tqdm
import torch

from coronal_diffusion.dataset import CoronalFieldDatasetHDF
from coronal_diffusion.constants import device
from coronal_diffusion.models import DiffusionModel
import config


def main():
    test_dataset = CoronalFieldDatasetHDF(config.test_dataset_path)
    model = DiffusionModel().to(device)

    radii = np.arange(
        config.min_radius, config.max_radius + config.res_radius, config.res_radius
    )

    # Get mean absolute values for asinh scalings
    total_unscaled_abs_value = np.zeros_like(radii)
    counter = 0
    
    for coeffs, _ in tqdm.tqdm(test_dataset):
        counter += 1

        #if counter > 100:
        #    break
        
        coeffs = coeffs.to(device)
        
        for i, r in enumerate(radii):
            img = make_img(model, coeffs, r)
            total_unscaled_abs_value[i] += torch.abs(img).mean().item()

    unscaled_abs_value = total_unscaled_abs_value / counter

    # Get the standard deviation after asinh scalings
    sum_sq = np.zeros_like(radii)
    total = np.zeros_like(radii)
    counter = 0
    
    for coeffs, _ in tqdm.tqdm(test_dataset):
        counter += 1
        
        #if counter > 100:
        #    break

        # mean is zero by construction
        coeffs = coeffs.to(device)

        for i, r in enumerate(radii):
            img = make_img(model, coeffs, r)
            img = torch.asinh(img / unscaled_abs_value[i])
            sum_sq[i] += torch.sum(img**2).item()
            total[i] += img.shape[0] * img.shape[1]

    std = np.sqrt(sum_sq / (total - 1))

    out = {"unscaled_abs": unscaled_abs_value.tolist(), "std": std.tolist()}

    with open(config.scalers_path, "w") as fh:
        json.dump(out, fh, indent=4)


def make_img(model, coeffs, r):
    radial_scaling = torch.zeros(coeffs.shape, dtype=torch.float32, device=device)

    for n in range(coeffs.shape[-1]):
        radial_scaling[n, :] = 1 / r ** (n + 1)

    img = model.isht(radial_scaling * coeffs).float()

    return img


if __name__ == "__main__":
    main()
    print(f"Scalers written to {config.scalers_path}")
