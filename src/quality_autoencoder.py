"""Residual quality-focused autoencoder for 128 x 128 RGB evidence images."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_NAME = "quality_residual_ae_v1"
LATENT_CHANNELS = 32
LATENT_SIZE = 16
COMPRESSION_RATIO = 6


class ResidualBlock(nn.Module):
    """Two-convolution residual block without batch-dependent normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class UpsampleBlock(nn.Module):
    """Resize then convolve to avoid transposed-convolution checkerboards."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.residual = ResidualBlock(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
        return self.residual(F.silu(self.conv(outputs), inplace=True))


class QualityAutoencoder(nn.Module):
    """Encode to 32 x 16 x 16 (6x compression) and reconstruct without skips."""

    architecture_name = ARCHITECTURE_NAME
    compression_ratio = COMPRESSION_RATIO
    latent_shape = (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, padding=2),
            nn.SiLU(inplace=True),
            ResidualBlock(48),
            nn.Conv2d(48, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResidualBlock(96),
            nn.Conv2d(96, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, LATENT_CHANNELS, kernel_size=1),
        )
        self.decoder_input = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.decoder = nn.Sequential(
            UpsampleBlock(128, 96),
            UpsampleBlock(96, 64),
            UpsampleBlock(64, 48),
            ResidualBlock(48),
            nn.Conv2d(48, 3, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the spatial bottleneck representation."""
        return self.encoder(inputs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a bottleneck tensor without encoder skip connections."""
        return self.decoder(self.decoder_input(latent))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct an input batch."""
        return self.decode(self.encode(inputs))


__all__ = [
    "ARCHITECTURE_NAME",
    "COMPRESSION_RATIO",
    "QualityAutoencoder",
]
