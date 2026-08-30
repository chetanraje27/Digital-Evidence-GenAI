"""Simple convolutional Autoencoder for 128 x 128 RGB CASIA images."""

from __future__ import annotations

import torch
from torch import nn


class ConvolutionalAutoencoder(nn.Module):
    """Encode an RGB image to 32 x 8 x 8 and reconstruct it.

    Each encoder layer halves width and height. The decoder mirrors this with
    transposed convolutions. No BatchNorm is used, keeping the architecture
    small and its behavior straightforward to explain.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 x 16 x 16 -> 32 x 8 x 8 latent representation
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            # 32 x 8 x 8 -> 64 x 16 x 16
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 x 64 x 64 -> 3 x 128 x 128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spatial latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct an RGB image from a latent tensor."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct an input batch."""
        return self.decode(self.encode(x))
