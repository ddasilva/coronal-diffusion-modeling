import models
import torch
import numpy as np
import json


def sample(
    nsteps=50,
    radio_flux=1,
    output_dim=8372,
    input_dim=8372,
    hidden_dim=8372,
    device=None,
    nmax=90,
    weights_file="diffusion_model.pth",
    return_history=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run denoising diffusion process nsteps times
    model = models.DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))

    noise = torch.normal(mean=0, std=1, size=(1, output_dim))
    input = noise.float().to(device)
    radio_flux = torch.from_numpy(np.array([radio_flux]).reshape(1, -1)).float().to(device)

    history = [input.clone()]

    with torch.no_grad():
        model.eval()
        for i in range(nsteps):
            # Linearly decreasing noise level from 1 to 0 over nsteps
            noise_level = 1.0 - i / max(1, nsteps - 1)
            input = model(input, noise_level, radio_flux)
            history.append(input.clone())

    history = [H.cpu().detach().squeeze().numpy() for H in history]
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
        H = np.zeros((nmax + 1, nmax + 1))
        G = np.zeros((nmax + 1, nmax + 1))

        G[np.triu_indices(nmax + 1)] = input_np[: input_np.shape[0] // 2]
        H[np.triu_indices(nmax + 1)] = input_np[input_np.shape[0] // 2 :]

        G = G.T
        H = H.T
        return_value.append((G, H))

    if return_history:
        return return_value

    return return_value[-1]
