import models
import torch
import numpy as np
import json
from constants import *

def sample(
    weights_file="diffusion_model.pth",
    radio_flux=0,
    model=None,
    output_dim=X_SIZE,
    input_dim=X_SIZE,
    hidden_dim=X_SIZE,
    device=None,
    nmax=90,
    sf=1,
    n=20,
    eta=0.1,
    return_history=False,
    method='ddim',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run deterministic diffusion process nsteps times
    if model is None:
        model = models.DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
        model.load_state_dict(torch.load(weights_file, map_location=device))

    radio_flux = torch.from_numpy(np.array([radio_flux]).reshape(1, -1)).float().to(device)
    
    if method == 'ddim':
        history = sample_ddim(model, radio_flux, sf, n, eta)
    elif method == 'ddpm':
        history = sample_ddpm(model, radio_flux, sf, n)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    if not return_history:
        history = [history[-1]]

    # Load scalers
    with open("scalers.json") as fh:
        scalers = json.load(fh)

    mean = np.array(scalers["mean"])
    std = np.array(scalers["std"])
    return_value = []

    for input_np in history:
        # Descale
        input_np = input_np * std + mean

        # Convert to G and H Matrices
        G = np.zeros((nmax + 1, nmax + 1))
        H = np.zeros((nmax + 1, nmax + 1))
        Htemp = np.zeros((nmax, nmax))

        cutoff = np.tril_indices(nmax + 1)[0].size

        G[np.tril_indices(nmax + 1)] = input_np[:cutoff]
        Htemp[np.tril_indices(nmax)] = input_np[cutoff:]

        H[1:, 1:] = Htemp

        return_value.append((G, H))

    if return_history:
        return return_value

    return return_value[-1]



# sample quickly using DDIM
@torch.no_grad()
def sample_ddim(model, radio_flux, sf, n, eta):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(1, X_SIZE).to(device)

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = timesteps // n
    for i in range(timesteps, 0, -step_size):
        #print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:].to(device)

        eps = model(samples, noise_level=t, radio_flux=radio_flux)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps, sf, eta)
        intermediate.append(samples.squeeze().detach().cpu().numpy())

    history = np.stack(intermediate)
    return history


# define sampling function for DDIM   
# removes the noise using ddim
def denoise_ddim(x, t, t_prev, pred_noise, sf, eta=0.0):
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
    ab = ab_t[t]
    ab_prev = ab_t[t_prev]
    
    # Predict x0 (original sample)
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - sf * (1 - ab).sqrt() * pred_noise)
    
    # Directional update
    dir_xt = sf * (1 - ab_prev).sqrt() * pred_noise
    
    # Add stochasticity
    if eta > 0:
        noise = torch.randn_like(x)  # Random noise
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        dir_xt += sigma * noise

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
    b = b_t[t_idx]
    a = a_t[t_idx]
    ab = ab_t[t_idx]
    ab_prev = ab_t[t_idx - 1]

    # Estimate the clean sample x0
    x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()

    # Compute posterior mean and variance
    mean = (ab_prev.sqrt() * b / (1 - ab)) * x0_pred + \
           ((1 - ab_prev) * a.sqrt() / (1 - ab)) * x

    # Sample from posterior distribution
    if t_idx > 1:
        noise = torch.randn_like(x)
        var = sf * ((1 - ab_prev) / (1 - ab)) * b
        return mean + var.sqrt() * noise
    else:
        return mean  # final step, no noise

@torch.no_grad()
def sample_ddpm(model, radio_flux, sf, n):
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
    x = torch.randn(1, X_SIZE, device=device)
    history = [x.squeeze().cpu().numpy()]

    for t_idx in reversed(range(1, timesteps + 1)):
        t = torch.tensor([t_idx / timesteps]).to(device)
        eps = model(x, noise_level=t, radio_flux=radio_flux)    # predict noise e_(x_t,t)
        x = denoise_ddpm(x, t_idx, eps, sf)
        history.append(x.squeeze().cpu().numpy())

    return np.stack(history, axis=0)
