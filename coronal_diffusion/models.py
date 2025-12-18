from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_harmonics as th
import math

import config
from config import nmax, nlon, nlat, radii


class DiffusionModel(nn.Module):
    """
    Main diffusion model wrapper that handles:
    - Spherical harmonic transforms (SHT/ISHT)
    - Multi-channel UNet for denoising
    - Radial channel management
    """

    def __init__(
        self,
        nmax: int = config.nmax,
        nlat: int = config.nlat,
        nlon: int = config.nlon,
    ):
        super().__init__()

        # Store configuration
        self.nmax = nmax
        self.nlat = nlat
        self.nlon = nlon

        # Compute radii grid
        self.radii = config.radii
        self.n_channels = len(self.radii)

        # Spherical harmonic transforms
        self.isht = th.InverseRealSHT(
            nlat=nlat,
            nlon=nlon,
            lmax=nmax + 1,
            mmax=nmax + 1,
            norm="ortho",
        )

        self.sht = th.RealSHT(
            nlat=nlat,
            nlon=nlon,
            lmax=nmax + 1,
            mmax=nmax + 1,
            norm="ortho",
        )

        # Main denoising network
        self.unet = MultispectralDiffusionUNet(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            base_channels=64,
            channel_multipliers=(1, 2, 4, 8),
            time_emb_dim=256,
            context_emb_dim=256,
        )

    def forward(
        self,
        img_with_noise: torch.Tensor,
        noise_level: torch.Tensor,
        radio_flux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise in the noisy image.

        Args:
            img_with_noise: Noisy images [B, n_channels, H, W]
            noise_level: Diffusion timestep normalized to [0, 1], shape [B] or [B, 1]
            radio_flux: Conditioning scalar (solar activity), shape [B] or [B, 1]

        Returns:
            Predicted noise [B, n_channels, H, W]
        """
        # Ensure proper shapes
        if noise_level.dim() == 2:
            noise_level = noise_level.squeeze(-1)
        if radio_flux.dim() == 1:
            radio_flux = radio_flux.unsqueeze(-1)

        return self.unet(img_with_noise, noise_level, radio_flux)


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for continuous values (time, context)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and context embeddings"""

    def __init__(self, in_channels, out_channels, emb_dim, dropout=0.1):
        super().__init__()

        # Combined embedding projection (time + context)
        self.emb_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_dim, out_channels * 2)  # *2 for scale and shift
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)

        # Add embedding with adaptive group normalization (scale and shift)
        emb_out = self.emb_mlp(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.activation(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing global dependencies"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Get Q, K, V
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(c // self.num_heads)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(b, c, h, w)

        # Project and residual
        out = self.proj(out)
        return x + out


class DownBlock(nn.Module):
    """Downsampling block with residual blocks"""

    def __init__(
        self, in_channels, out_channels, emb_dim, downsample=True, use_attention=False
    ):
        super().__init__()
        self.resblock1 = ResidualBlock(in_channels, out_channels, emb_dim)
        self.resblock2 = ResidualBlock(out_channels, out_channels, emb_dim)

        self.attention = (
            AttentionBlock(out_channels) if use_attention else nn.Identity()
        )

        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x, emb):
        x = self.resblock1(x, emb)
        x = self.resblock2(x, emb)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with residual blocks and skip connections"""

    def __init__(
        self, in_channels, out_channels, emb_dim, upsample=True, use_attention=False
    ):
        super().__init__()
        self.resblock1 = ResidualBlock(in_channels, out_channels, emb_dim)
        self.resblock2 = ResidualBlock(out_channels, out_channels, emb_dim)

        self.attention = (
            AttentionBlock(out_channels) if use_attention else nn.Identity()
        )

        self.upsample = upsample
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, skip, emb):
        # Resize x to match skip if needed (handles odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.resblock1(x, emb)
        x = self.resblock2(x, emb)
        x = self.attention(x)

        if self.upsample:
            x = self.upsample_conv(x)

        return x


class MultispectralDiffusionUNet(nn.Module):
    """
    U-Net for multispectral image denoising with scalar context conditioning
    Input shape: (B, 16, 180, 90) - 16 channels, 180 width, 90 height
    Context: (B,) - scalar value per sample (already normalized)
    """

    def __init__(
        self,
        in_channels=16,
        out_channels=16,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        time_emb_dim=256,
        context_emb_dim=256,
        use_attention_levels=(False, False, True, True),
        dropout=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Combined embedding dimension
        combined_emb_dim = time_emb_dim + context_emb_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Context embedding (for scalar values)
        self.context_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(context_emb_dim),
            nn.Linear(context_emb_dim, context_emb_dim),
            nn.SiLU(),
            nn.Linear(context_emb_dim, context_emb_dim),
        )

        # Combined embedding projection
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_emb_dim, combined_emb_dim),
            nn.SiLU(),
            nn.Linear(combined_emb_dim, combined_emb_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Calculate channel sizes
        channels = [base_channels * m for m in channel_multipliers]

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    combined_emb_dim,
                    downsample=(i < len(channels) - 1),
                    use_attention=use_attention_levels[i],
                )
            )
            in_ch = out_ch

        # Bottleneck
        self.bottleneck_res1 = ResidualBlock(
            channels[-1], channels[-1], combined_emb_dim
        )
        self.bottleneck_attn = AttentionBlock(channels[-1])
        self.bottleneck_res2 = ResidualBlock(
            channels[-1], channels[-1], combined_emb_dim
        )

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            out_ch = (
                reversed_channels[i + 1]
                if i < len(reversed_channels) - 1
                else base_channels
            )

            self.up_blocks.append(
                UpBlock(
                    in_ch * 2,  # *2 for skip connection
                    out_ch,
                    combined_emb_dim,
                    upsample=(i < len(reversed_channels) - 1),
                    use_attention=use_attention_levels[-(i + 1)],
                )
            )

        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, context):
        """
        Args:
            x: Input tensor of shape (B, 16, 180, 90)
            timesteps: Timesteps tensor of shape (B,)
            context: Scalar context tensor of shape (B,) or (B, 1) - already normalized

        Returns:
            Denoised output of shape (B, 16, 180, 90)
        """
        # Handle context shape
        if context.dim() == 2:
            context = context.squeeze(-1)

        # Get embeddings
        time_emb = self.time_embed(timesteps)
        context_emb = self.context_embed(context)

        # Combine embeddings
        combined_emb = torch.cat([time_emb, context_emb], dim=-1)
        combined_emb = self.combined_mlp(combined_emb)

        # Initial conv
        x = self.conv_in(x)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, combined_emb)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck_res1(x, combined_emb)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_res2(x, combined_emb)

        # Decoder
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, combined_emb)

        # Final conv
        x = self.conv_out(x)

        return x
