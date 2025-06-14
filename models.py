import torch
import torch.nn as nn
import numpy as np



class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, noise_level, radio_flux=None):
        # Add noise to the input (forward process)
        noise = torch.normal(mean=0, std=noise_level, size=x.size()).to(x.device)
        noisy_x = x + noise

        # Encode and decode (reverse process)
        encoded = self.encoder(noisy_x)
        middle = torch.cat((encoded + noisy_x, radio_flux), dim=1) 
        reconstructed = self.decoder(middle)

        # Additive skip connection
        output = reconstructed

        return output
