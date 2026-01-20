from dataclasses import dataclass
import json
import itertools

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


from coronal_diffusion import constants, models
from coronal_diffusion.constants import device, N_CONTEXT
from coronal_diffusion.utils import flat_to_GH
import config


@dataclass
class SamplingData:
    std: np.array  # scaling std (after asinh scaling)
    unscaled_abs: np.array  # potential image asinh scaling factor
    A: torch.Tensor  # spherical harmonic fit array


def load_sampling_data():
    # Load spherical harmonic fitting matrix
    hdf = h5py.File(config.spharm_fit_mat_path)
    A = torch.tensor(hdf["A"][:], device=device)
    hdf.close()

    # Load scalers parameters
    with open(config.scalers_path) as fh:
        scalers_dict = json.load(fh)

    std = np.array(scalers_dict["std"])
    unscaled_abs = np.array(scalers_dict["unscaled_abs"])

    # Load spherical harmonic fitting matrix
    return SamplingData(std=std, unscaled_abs=unscaled_abs, A=A)


def sample(
    sampling_data,
    weights_file=None,
    model=None,
    context=(0.0, 400, 0.0),
    n=20,
    eta=0.1,
    method='ddpm',
    return_history=False,
):
    # Check command line arguments
    assert weights_file or model, "Must provide either weights_file= or model="

    # Load model if not already provided
    if model is None:
        model = models.DiffusionModel().to(constants.device)
        model.load_state_dict(torch.load(weights_file, map_location=constants.device))
        model.eval()

    # Run deterministic diffusion process nsteps times
    context = (
        torch.from_numpy(np.array(context).reshape(1, N_CONTEXT))
        .float()
        .to(constants.device)
    )

    # Call sample_ddim
    if method == 'ddim':
        imgs, history_pred_noise = sample_ddim(model, context, n, eta)
    elif method == 'ddpm':
        imgs = sample_ddpm(model, context)
        
    # Perform sphericla harmonic fit of last image
    G, H = spherical_harm_fit_smaller(imgs[-1], sampling_data)

    # Branch based on return_history
    if return_history:
        return imgs, (G, H)
    else:
        return imgs[-1], (G, H)


def spherical_harm_fit_smaller(img, sampling_data, fitrad=config.radii.size):
    # Rescale image -------------------------------------
    img_rescaled = np.zeros(img.shape)
    nrad, nlat, nlon = img.shape

    for i in range(fitrad):
        img_rescaled[i] = (
            np.sinh(img[i] * sampling_data.std[i]) * sampling_data.unscaled_abs[i]
        )

    # Build right hand side of Ax=b equation ------------------
    b = torch.zeros(fitrad * nlat * nlon, device=device)

    for i in range(fitrad):
        start_row = i * nlat * nlon
        end_row = start_row + nlat * nlon
        b[start_row:end_row] = torch.tensor(img_rescaled[i].flatten(), device=device)

    # New approcah
    print(f"Fitting spherical harmonics ({config.fit_nmax})")
    X = np.zeros(sampling_data.A.shape[1] + 1)
    X[1:] = (
        torch.linalg.lstsq(sampling_data.A[: fitrad * nlat * nlon, :], b)[0]
        .cpu()
        .numpy()
    )

    G, H = flat_to_GH(X, nmax=config.fit_nmax)

    return G, H


@torch.no_grad()
def sample_ddim(model, context, n, eta):
    # sample initial noise
    nrad = config.radii.size
    shape = (1, nrad, config.nlat, config.nlon)
    img = torch.randn(shape, device=constants.device)

    # Loop
    step_size = constants.timesteps // n
    img_all = [img.squeeze().detach().cpu().numpy()]
    pred_noise_all = []

    for i in range(constants.timesteps, 0, -step_size):
        t = torch.tensor([i / constants.timesteps]).to(constants.device)

        pred_noise = model(img, noise_level=t, context=context)
        pred_noise_all.append(pred_noise.squeeze().detach().cpu().numpy())

        print(f"\rDenoising Step {i}", end="")

        img = denoise_ddim(img, i, i - step_size, pred_noise, eta)
        img_all.append(img.squeeze().detach().cpu().numpy())

    print()

    history = np.stack(img_all)
    history_pred_noise = np.stack(pred_noise_all)

    return history, history_pred_noise


# define sampling function for DDIM
# removes the noise using ddim
def denoise_ddim(x, t, t_prev, pred_noise, eta):
    """
    Denoise using DDIM with optional stochasticity controlled by eta.

    Args:
        x (torch.Tensor): Current noisy sample.
        t (int): Current timestep.
        t_prev (int): Previous timestep.
        pred_noise (torch.Tensor): Predicted noise.
        sf (float): Scaling factor for noise.
        eta (float): Stochasticity parameter (0 for deterministic, >0 for stochastic).

    Returns:
        torch.Tensor: Denoised sample.
    """
    ab = constants.ab_t[t]
    ab_prev = constants.ab_t[t_prev]

    # Predict x0 (original sample)
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)

    # Directional update
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    # Add stochasticity
    if eta > 0:
        shape = x.shape
        img_noise = torch.randn(shape).to(constants.device)

        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        dir_xt += sigma * img_noise

    return x0_pred + dir_xt


@torch.no_grad()
def denoise_ddpm(x, t_idx, eps):
    """
    DDPM denoising step based on Ho et al. (2020).

    Args:
        x (torch.Tensor): Current noisy sample at timestep t.
        t_idx (int): Current timestep index.
        eps (torch.Tensor): Model's predicted noise (epsilon).

    Returns:
        torch.Tensor: Sample at timestep t-1.
    """
    b = constants.b_t[t_idx]
    a = constants.a_t[t_idx]
    ab = constants.ab_t[t_idx]
    ab_prev = constants.ab_t[t_idx - 1]

    # Estimate the clean sample x0
    x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()

    # Compute posterior mean and variance
    mean = (ab_prev.sqrt() * b / (1 - ab)) * x0_pred + (
        (1 - ab_prev) * a.sqrt() / (1 - ab)
    ) * x

    # Sample from posterior distribution
    if t_idx > 1:
        noise = torch.randn_like(x)
        var = ((1 - ab_prev) / (1 - ab)) * b
        return mean + var.sqrt() * noise
    else:
        return mean  # final step, no noise


@torch.no_grad()
def sample_ddpm(model, context):
    """
    DDPM ancestral sampling (standard diffusion sampling, not deterministic like DDIM).
    Args:
        model: The trained diffusion model.
        context: Conditioning variable.
        n: Number of steps to sample.
    Returns:
        history: np.ndarray of generated samples at each step.
    """
    nrad = config.radii.size
    shape = (1, nrad, config.nlat, config.nlon)
    x = torch.randn(shape, device=constants.device)
    history = [x.squeeze().cpu().numpy()]

    for t_idx in reversed(range(1, constants.timesteps + 1)):
        print(f"\rDenoising Step {t_idx}              ", end="")
        t = torch.tensor([t_idx / constants.timesteps]).to(constants.device)
        eps = model(x, noise_level=t, context=context)  # predict noise e_(x_t,t)
        x = denoise_ddpm(x, t_idx, eps)
        history.append(x.squeeze().cpu().numpy())
    print()
    return np.stack(history, axis=0)
