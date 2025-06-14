import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, noise_level):
        # Add noise to the input (forward process)
        noise = torch.randn_like(x) * noise_level
        noisy_x = x + noise

        # Encode and decode (reverse process)
        encoded = self.encoder(noisy_x)
        reconstructed = self.decoder(encoded + noisy_x)

        # Additive skip connection
        output = reconstructed

        return output
