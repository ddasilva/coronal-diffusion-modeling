import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_harmonics as th
import math

from coronal_diffusion.constants import asinh_sf
import config
from config import nmax


class DiffusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        nlat = 90
        nlon = 180
        self.isht = th.InverseRealSHT(
            nlat=nlat, nlon=nlon, lmax=config.nmax + 1, norm="ortho"
        )
        self.sht = th.RealSHT(nlat=nlat, nlon=nlon, lmax=config.nmax + 1, norm="ortho")
        self.unet = DiffusionUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            channel_multipliers=(1, 2, 4, 8),
            # channel_multipliers=(1, 2, 4),
            num_res_blocks=2,
            attention_levels=(2, 3),
        )

    def forward(self, img_with_noise, noise_level, radio_flux, return_noise=False):
        img_noise = self.unet(img_with_noise[:, None, :, :], noise_level, radio_flux)
        img_noise = img_noise[:, 0]

        return img_noise


class CircularConv2d(nn.Module):
    """2D Convolution with circular padding on longitude (width) dimension"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        # Circular padding on width (longitude), regular padding on height (latitude)
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        return self.conv(x)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion timestep"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)

        # Attention
        attn = torch.softmax(
            q @ k.transpose(-2, -1) / math.sqrt(C // self.num_heads), dim=-1
        )
        h = (attn @ v).transpose(2, 3).reshape(B, C, H, W)

        h = self.proj(h)
        return x + h


class Downsample(nn.Module):
    """Downsampling with circular padding on longitude"""

    def __init__(self, channels):
        super().__init__()
        self.conv = CircularConv2d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling with circular convolution"""

    def __init__(self, channels):
        super().__init__()
        self.conv = CircularConv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """
    UNet for diffusion model on 90x180 (lat, lon) images

    Args:
        in_channels: Number of input image channels
        out_channels: Number of output image channels
        base_channels: Base number of channels (default: 64)
        channel_multipliers: Channel multipliers for each resolution level
        num_res_blocks: Number of residual blocks per level
        attention_levels: Which levels to apply attention (e.g., [1, 2])
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.conv_in = CircularConv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        # Encoder
        self.encoder = nn.ModuleList()
        channels_list = [base_channels]
        in_ch = base_channels

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResidualBlock(in_ch, out_ch, time_emb_dim, dropout)]

                if level in attention_levels:
                    layers.append(AttentionBlock(out_ch))

                self.encoder.append(nn.ModuleList(layers))
                in_ch = out_ch
                channels_list.append(in_ch)

            if level != len(channel_multipliers) - 1:
                self.encoder.append(nn.ModuleList([Downsample(in_ch)]))
                channels_list.append(in_ch)

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [
                ResidualBlock(in_ch, in_ch, time_emb_dim, dropout),
                AttentionBlock(in_ch),
                ResidualBlock(in_ch, in_ch, time_emb_dim, dropout),
            ]
        )

        # Decoder
        self.decoder = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = channels_list.pop()
                layers = [ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout)]

                if level in attention_levels:
                    layers.append(AttentionBlock(out_ch))

                self.decoder.append(nn.ModuleList(layers))
                in_ch = out_ch

            if level != 0:
                self.decoder.append(nn.ModuleList([Upsample(in_ch)]))

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            CircularConv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, img, t, scalar_context):
        """
        Args:
            img: Input image tensor [B, C, H, W] where H=90, W=180
            t: Diffusion timestep [B] or [B, 1]
            scalar_context: Scalar conditioning [B] or [B, 1]

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Ensure proper shapes
        if img.dim() != 4:
            raise ValueError(
                f"Expected img to be 4D [B, C, H, W], got shape {img.shape}"
            )

        # Ensure t is 1D
        if t.dim() > 1:
            t = t.squeeze(-1)

        # Ensure scalar_context is [B, 1]
        if scalar_context.dim() == 1:
            scalar_context = scalar_context.unsqueeze(-1)

        # Time embedding
        t_emb = self.time_embedding(t)

        # Initial convolution
        x = self.conv_in(img)

        # Encoder with skip connections
        skips = [x]

        for module_list in self.encoder:
            for module in module_list:
                if isinstance(module, ResidualBlock):
                    x = module(x, t_emb, scalar_context)
                elif isinstance(module, AttentionBlock):
                    x = module(x)
                elif isinstance(module, Downsample):
                    x = module(x)
            skips.append(x)

        # Bottleneck
        for module in self.bottleneck:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb, scalar_context)
            else:
                x = module(x)

        # Decoder with skip connections
        for module_list in self.decoder:
            for module in module_list:
                if isinstance(module, ResidualBlock):
                    skip = skips.pop()

                    # Match spatial dimensions if they don't align (due to odd dimensions)
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

                    x = torch.cat([x, skip], dim=1)
                    x = module(x, t_emb, scalar_context)
                elif isinstance(module, AttentionBlock):
                    x = module(x)
                elif isinstance(module, Upsample):
                    x = module(x)

        # Output
        x = self.conv_out(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        # FIX: Use adaptive number of groups
        num_groups = min(8, in_channels)  # Don't exceed channel count
        if in_channels % num_groups != 0:
            # Find largest divisor
            for ng in range(num_groups, 0, -1):
                if in_channels % ng == 0:
                    num_groups = ng
                    break

        self.conv1 = CircularConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CircularConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        self.scalar_mlp = nn.Sequential(nn.SiLU(), nn.Linear(1, out_channels))

        # Fixed normalization
        self.norm1 = nn.GroupNorm(min(8, in_channels // 4), in_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels // 4), out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = CircularConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb, scalar_context):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time conditioning
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb

        # Add scalar context conditioning
        scalar_emb = self.scalar_mlp(scalar_context)[:, :, None, None]
        h = h + scalar_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


def get_num_groups(channels, target_groups=8):
    """Get valid number of groups for GroupNorm"""
    if channels < target_groups:
        return channels

    # Find largest divisor <= target_groups
    for ng in range(target_groups, 0, -1):
        if channels % ng == 0:
            return ng
    return 1
