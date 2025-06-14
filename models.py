import torch
import torch.nn as nn
import numpy as np



class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        # Add more layers and normalization for better expressiveness and stability
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.LeakyReLU(0.1)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.act3 = nn.LeakyReLU(0.1)

        # Project radio_flux to hidden_dim and add as context
        self.context_proj = nn.Linear(1, hidden_dim)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        # For input-output dimension match in residual
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x, noise_level, radio_flux=None):
        # Add noise to the input (forward process)
        noise = torch.normal(mean=torch.zeros_like(noise_level), std=noise_level).to(x.device)
        noisy_x = x + noise

        # First layer
        out = self.input_layer(noisy_x)
        out = self.bn1(out)
        out = self.act1(out)
        # Residual connection from input
        res = self.residual_proj(noisy_x)
        out = out + res

        out = self.hidden1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.hidden2(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.dropout(out)

        # Noise embedding
        if isinstance(noise_level, float) or len(noise_level.shape) == 0:
            noise_level = torch.tensor([noise_level], dtype=out.dtype, device=out.device).repeat(out.shape[0], 1)
        elif len(noise_level.shape) == 1:
            noise_level = noise_level.unsqueeze(1)
        noise_emb = self.noise_embed(noise_level)
        out = out + noise_emb
        
        # Context embedding
        if radio_flux is not None:
            context = self.context_proj(radio_flux)
            out = out + context
        out = self.final(out)

        # Additive skip connection
        output = out

        return output
