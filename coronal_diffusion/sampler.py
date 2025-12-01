import json

import torch
import numpy as np

from coronal_diffusion import constants, models
from coronal_diffusion.constants import asinh_sf
from coronal_diffusion.utils import flat_to_GH
import config


def sample(
    model=None,
    weights_file=None,
    radio_flux=0,
    nmax=config.nmax,
    sf=1,
    n=20,
    eta=0.1,
    return_history=False,
    method="ddim",
):
    # Run deterministic diffusion process nsteps times
    if model is None:
        model = models.DiffusionModel().to(constants.device)
        model.load_state_dict(torch.load(weights_file, map_location=constants.device))

    radio_flux = (
        torch.from_numpy(np.array([radio_flux]).reshape(1, -1))
        .float()
        .to(constants.device)
    )

    # Load scalers
    with open(config.scalers_path) as fh:
        scalers = json.load(fh)

    stdG, stdH = flat_to_GH(np.array(scalers["std"]))
    stdG = torch.tensor(stdG)
    stdH = torch.tensor(stdH)
    std = stdG, stdH

    if method == "ddim":
        history, history_eps = sample_ddim(model, radio_flux, sf, n, eta, std)
    elif method == "ddpm":
        history = sample_ddpm(model, radio_flux, sf, n, seed_mean, seed_std)
        history_eps = None
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    if not return_history:
        history = [history[-1]]

    return_value = []

    for img in history:
        img_scaled = np.sinh(img) * sf
        coeffs = (
            model.sht(torch.tensor(img_scaled, device=constants.device)).cpu().numpy()
        )

        # Convert to G and H Matrices
        G = coeffs.real
        H = coeffs.imag

        return_value.append((img, (G, H)))

    if return_history:
        return return_value

    return return_value[-1]


# sample quickly using DDIM
@torch.no_grad()
def sample_ddim(model, radio_flux, sf, n, eta, std):
    # sample initial noise
    shape = (1, config.nmax + 1, config.nmax + 1)
    coeffs = torch.zeros(shape, dtype=torch.complex64)
    coeffs += (
        torch.randn(shape) * torch.asinh(std[0] / asinh_sf)
        + torch.randn(shape) * torch.asinh(std[1] / asinh_sf) * 1j
    )
    coeffs = torch.sinh(coeffs) * asinh_sf
    coeffs = coeffs.to(constants.device)

    img = model.isht(coeffs)
    img = torch.asinh(img / asinh_sf)

    # Loop
    step_size = constants.timesteps // n
    img_all = [img.squeeze().detach().cpu().numpy()]
    eps_all = []

    for i in range(constants.timesteps, 0, -step_size):
        t = torch.tensor([i / constants.timesteps])[:].to(constants.device)

        eps = model(img, noise_level=t, radio_flux=radio_flux, return_noise=True)
        eps_all.append(eps.squeeze().detach().cpu().numpy())

        img = denoise_ddim(img, i, i - step_size, eps, sf, eta, std, model)
        img_all.append(img.squeeze().detach().cpu().numpy())
        # img_all.append(model.sht(img_rescaled).squeeze().detach().cpu().numpy())

    history = np.stack(img_all)
    history_eps = np.stack(eps_all)

    return history, history_eps


# define sampling function for DDIM
# removes the noise using ddim
def denoise_ddim(x, t, t_prev, pred_noise, sf, eta, std, model):
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
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - sf * (1 - ab).sqrt() * pred_noise)

    # Directional update
    dir_xt = sf * (1 - ab_prev).sqrt() * pred_noise

    # Add stochasticity
    if eta > 0:
        shape = (1, config.nmax + 1, config.nmax + 1)
        noise = torch.zeros(shape, dtype=torch.complex64)
        noise += (
            torch.randn(shape) * torch.asinh(std[0] / asinh_sf)
            + torch.randn(shape) * torch.asinh(std[1] / asinh_sf) * 1j
        )
        noise = torch.sinh(noise) * asinh_sf
        noise = noise.to(constants.device)

        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        
        img_noise = model.isht(sigma * noise)
        img_noise = torch.asinh(img_noise / asinh_sf)
        dir_xt += img_noise

        #sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        #dir_xt += sigma * img_noise

    return x0_pred + dir_xt


@torch.no_grad()
def denoise_ddpm(x, t_idx, eps, sf):
    """
    DDPM denoising step based on Ho et al. (2020).

    Args:
        x (torch.Tensor): Current noisy sample at timestep t.
        t_idx (int): Current timestep index.
        eps (torch.Tensor): Model's predicted noise (epsilon).
        sf (float): Noise scaling factor (1.0 for DDPM).

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
        var = sf * ((1 - ab_prev) / (1 - ab)) * b
        return mean + var.sqrt() * noise
    else:
        return mean  # final step, no noise


@torch.no_grad()
def sample_ddpm(model, radio_flux, sf, n, seed_mean, seed_std):
    """
    DDPM ancestral sampling (standard diffusion sampling, not deterministic like DDIM).
    Args:
        model: The trained diffusion model.
        radio_flux: Conditioning variable.
        sf: Scaling factor for noise.
        n: Number of steps to sample.
    Returns:
        history: np.ndarray of generated samples at each step.
    """
    x = torch.randn(1, config.X_SIZE).to(constants.device) * seed_std + seed_mean
    history = [x.squeeze().cpu().numpy()]

    for t_idx in reversed(range(1, constants.timesteps + 1)):
        t = torch.tensor([t_idx / constants.timesteps]).to(constants.device)
        eps = model(x, noise_level=t, radio_flux=radio_flux)  # predict noise e_(x_t,t)
        x = denoise_ddpm(x, t_idx, eps, sf)
        history.append(x.squeeze().cpu().numpy())

    return np.stack(history, axis=0)
