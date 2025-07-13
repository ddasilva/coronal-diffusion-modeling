import torch
import torch.nn as nn
import numpy as np
import torch_harmonics as th
import math

def fix_coeffs(G, H):
    """
    Convert real spherical harmonics coefficients (G and H) to complex form (C and S).
    
    Args:
        G (torch.Tensor): Real spherical harmonics coefficients G_l^m
                         Shape: (max_degree + 1, max_degree + 1) or flattened
        H (torch.Tensor): Real spherical harmonics coefficients H_l^m  
                         Shape: (max_degree + 1, max_degree + 1) or flattened
    
    Returns:
        tuple: (C, S) where:
            C (torch.Tensor): Complex coefficients for positive m
            S (torch.Tensor): Complex coefficients for negative m
    
    Note:
        The conversion follows the standard convention where:
        - For m > 0: C_l^m = (G_l^m - i*H_l^m) / sqrt(2)
        - For m < 0: S_l^|m| = (G_l^|m| + i*H_l^|m|) / sqrt(2)  
        - For m = 0: C_l^0 = G_l^0 (purely real)
    """
    
    # Ensure inputs are torch tensors
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, dtype=torch.float32)
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
    
    # Get the maximum degree from the shape
    if G.dim() == 2:
        max_degree = G.shape[0] - 1
    else:
        # Assume flattened format and infer max degree
        max_degree = int(np.sqrt(len(G))) - 1
        G = G.reshape(max_degree + 1, max_degree + 1)
        H = H.reshape(max_degree + 1, max_degree + 1)
    
    # Initialize complex coefficient arrays
    C = torch.zeros((max_degree + 1, max_degree + 1), dtype=torch.complex64)
    S = torch.zeros((max_degree + 1, max_degree + 1), dtype=torch.complex64)
    
    sqrt2 = torch.sqrt(torch.tensor(2.0))
    
    for l in range(max_degree + 1):
        for m in range(l + 1):
            if m == 0:
                # For m = 0, coefficient is purely real
                C[l, 0] = G[l, 0].to(torch.complex64)
            else:
                # For m > 0: C_l^m = (G_l^m - i*H_l^m) / sqrt(2)
                C[l, m] = (G[l, m] - 1j * H[l, m]) / sqrt2
                
                # For m < 0: S_l^m = (G_l^m + i*H_l^m) / sqrt(2)
                # Store at positive index for convenience
                S[l, m] = (G[l, m] + 1j * H[l, m]) / sqrt2
    
    return C, S


def fix_coeffs_batch(G, H):
    """
    Batch-vectorized conversion of real spherical harmonics coefficients (G and H) to a single set of complex coefficients.
    
    Args:
        G (torch.Tensor): Real spherical harmonics coefficients G_l^m
                         Shape: (batch_size, max_degree + 1, max_degree + 1) or 
                                (batch_size, flattened_size)
        H (torch.Tensor): Real spherical harmonics coefficients H_l^m  
                         Shape: (batch_size, max_degree + 1, max_degree + 1) or
                                (batch_size, flattened_size)
    
    Returns:
        torch.Tensor: Complex coefficients combining G and H
                      Shape: (batch_size, max_degree + 1, max_degree + 1)
    """
    # Ensure inputs are torch tensors
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, dtype=torch.float32)
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
    
    # Handle different input shapes
    if G.dim() == 3:
        batch_size, max_degree_plus_1, _ = G.shape
        max_degree = max_degree_plus_1 - 1
    elif G.dim() == 2:
        batch_size = G.shape[0]
        flattened_size = G.shape[1]
        max_degree = int(np.sqrt(flattened_size)) - 1
        max_degree_plus_1 = max_degree + 1
        
        # Reshape to (batch_size, max_degree + 1, max_degree + 1)
        G = G.reshape(batch_size, max_degree_plus_1, max_degree_plus_1)
        H = H.reshape(batch_size, max_degree_plus_1, max_degree_plus_1)
    else:
        raise ValueError(f"Expected G to have 2 or 3 dimensions, got {G.dim()}")
    
    device = G.device
    
    # Create masks for different regions (same for all batches)
    l_indices, m_indices = torch.meshgrid(
        torch.arange(max_degree + 1, device=device),
        torch.arange(max_degree + 1, device=device),
        indexing='ij'
    )
    
    # Valid coefficient mask: m <= l
    valid_mask = m_indices <= l_indices
    
    # Initialize complex coefficient array for the entire batch
    coeffs = torch.zeros((batch_size, max_degree + 1, max_degree + 1), dtype=torch.complex64, device=device)
    
    # Handle m = 0 case (purely real) - vectorized across batch
    coeffs[:, valid_mask & (m_indices == 0)] = G[:, valid_mask & (m_indices == 0)].to(torch.complex64)
    
    # Handle m > 0 case vectorized across batch
    # coeffs_l^m = G_l^m - i*H_l^m
    coeffs[:, valid_mask & (m_indices > 0)] = G[:, valid_mask & (m_indices > 0)] - 1j * H[:, valid_mask & (m_indices > 0)]
    
    return coeffs

# Alternative version with explicit broadcasting (more memory efficient for very large batches)
def fix_coeffs_batch_broadcast(G, H):
    """
    Memory-efficient batch-vectorized version using explicit broadcasting.
    """
    
    # Ensure inputs are torch tensors
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, dtype=torch.float32)
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
    
    # Handle different input shapes
    if G.dim() == 3:
        batch_size, max_degree_plus_1, _ = G.shape
        max_degree = max_degree_plus_1 - 1
    elif G.dim() == 2:
        batch_size = G.shape[0]
        flattened_size = G.shape[1]
        max_degree = int(np.sqrt(flattened_size)) - 1
        max_degree_plus_1 = max_degree + 1
        G = G.reshape(batch_size, max_degree_plus_1, max_degree_plus_1)
        H = H.reshape(batch_size, max_degree_plus_1, max_degree_plus_1)
    else:
        raise ValueError(f"Expected G to have 2 or 3 dimensions, got {G.dim()}")
    
    device = G.device
    
    # Create index tensors
    l_indices = torch.arange(max_degree + 1, device=device).unsqueeze(1)  # (max_degree+1, 1)
    m_indices = torch.arange(max_degree + 1, device=device).unsqueeze(0)  # (1, max_degree+1)
    
    # Create masks
    valid_mask = m_indices <= l_indices
    m_zero_mask = (m_indices == 0) & valid_mask
    m_positive_mask = (m_indices > 0) & valid_mask
    
    # Initialize output tensors
    C = torch.zeros_like(G, dtype=torch.complex64)
    S = torch.zeros_like(G, dtype=torch.complex64)
    
    # Apply transformations with broadcasting
    # m = 0 case
    C = torch.where(m_zero_mask.unsqueeze(0), G.to(torch.complex64), C)
    
    # m > 0 case
    complex_transform_c = (G - 1j * H) 
    complex_transform_s = (G + 1j * H) 
    
    C = torch.where(m_positive_mask.unsqueeze(0), complex_transform_c, C)
    S = torch.where(m_positive_mask.unsqueeze(0), complex_transform_s, S)
    
    return C, S


class MagnetogramModel(nn.Module):

    def __init__(self):
        super(MagnetogramModel, self).__init__()

        self.nlat = 180
        self.nlon = 360
        self.nmax = 90
        self.cutoff = np.tril_indices(self.nmax + 1)[0].size
        self.isht = th.InverseRealSHT(self.nlat, self.nlon, grid='equiangular', norm='ortho', lmax=self.nmax + 1, mmax=self.nmax + 1)

    def forward(self, out, r = 1.05):
        batch_size = out.shape[0]
        G = torch.zeros((batch_size, self.nmax + 1, self.nmax + 1)).to(out.device)
        H = torch.zeros((batch_size, self.nmax + 1, self.nmax + 1)).to(out.device)

        for i in range(batch_size):
            G[i][np.tril_indices(self.nmax + 1)] = out[i, :self.cutoff]
            H[i][1:, 1:][np.tril_indices(self.nmax)] = out[i, self.cutoff:]

        # Scale coefficients
        n = torch.arange(self.nmax + 1, dtype=torch.float32, device=out.device).view(-1, 1)
        scale = (n + 1).repeat(1, self.nmax + 1)
        G = G * scale
        H = H * scale

        coeffs = fix_coeffs_batch(G, H)

        # Evaluate at r = 1.05
        coeffs = coeffs * (1 / r**(n + 1))

        magnetogram = self.isht(coeffs)

        return magnetogram


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.LeakyReLU(0.1)
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
        out = self.hidden2(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.final(out)

        return out
