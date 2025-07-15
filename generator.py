import models
import torch
import numpy as np
import json
from constants import *

def sample(
    weights_file="diffusion_model.pth",
    radio_flux=0,
    model=None,
    output_dim=8281,
    input_dim=8281,
    hidden_dim=8281,
    device=None,
    nmax=90,
    sf=1,
    n=20,
    eta=0.0,
    return_history=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run deterministic diffusion process nsteps times
    if model is None:
        model = models.DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
        model.load_state_dict(torch.load(weights_file, map_location=device))

    radio_flux = torch.from_numpy(np.array([radio_flux]).reshape(1, -1)).float().to(device)
    history = sample_ddim(model, radio_flux, sf, n, eta)
    
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
    samples = torch.randn(1, 8281).to(device)

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = timesteps // n
    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

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