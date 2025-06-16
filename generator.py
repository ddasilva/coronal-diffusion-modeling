import models
import torch
import numpy as np
import json


def sample(
    weights_file="diffusion_model.pth",
    nsteps=50,
    radio_flux=1,
    output_dim=8281,
    input_dim=8281,
    hidden_dim=8281,
    device=None,
    nmax=90,
    return_history=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run denoising diffusion process nsteps times
    model = models.DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))

    input = torch.normal(mean=0, std=1, size=(1, output_dim))
    input = input.float().to(device)
    radio_flux = torch.from_numpy(np.array([radio_flux]).reshape(1, -1)).float().to(device)

    history = [input.clone()]

    with torch.no_grad():
        model.eval()
        for i in range(nsteps):
            # Linearly decreasing noise level from 1 to 0 over nsteps
            noise_level = torch.from_numpy(np.array([1.0 - i / max(1, nsteps - 1)])).float().to(device)
            pred_noise = model(input, noise_level, radio_flux)
            input -= pred_noise

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
