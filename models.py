import torch
import torch.nn as nn
import numpy as np


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.final = nn.Linear(hidden_dim, output_dim)
        self.context_proj = nn.Linear(1, hidden_dim)
        self.noise_embed = nn.Linear(1, hidden_dim)        
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, noisy_x, noise_level, radio_flux=None):
        out = self.input_layer(noisy_x)
        out = self.ln1(out)
        out = self.act1(out)
        res = self.residual_proj(noisy_x)
        out = out + res
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
        return out
