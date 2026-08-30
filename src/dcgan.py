"""DCGAN architecture for generating 64 x 64 RGB CASIA-style images."""

from __future__ import annotations

import torch
from torch import nn


class DCGANGenerator(nn.Module):
    """Map latent noise [N, 100, 1, 1] to RGB images [N, 3, 64, 64]."""

    def __init__(self, latent_dim: int = 100, image_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            # N x latent_dim x 1 x 1 -> N x 512 x 4 x 4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 4 x 4 -> 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class DCGANDiscriminator(nn.Module):
    """Return one unbounded real/fake logit per 64 x 64 RGB image."""

    def __init__(self, image_channels: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # 3 x 64 x 64 -> 64 x 32 x 32 (no BatchNorm in first block)
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8 -> 512 x 4 x 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4 -> one real/fake logit
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def initialize_dcgan_weights(module: nn.Module) -> None:
    """Apply the standard DCGAN normal initialization in-place."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def build_dcgan(
    latent_dim: int = 100, image_channels: int = 3
) -> tuple[DCGANGenerator, DCGANDiscriminator]:
    """Construct and initialize a matching generator/discriminator pair."""
    generator = DCGANGenerator(latent_dim, image_channels)
    discriminator = DCGANDiscriminator(image_channels)
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)
    return generator, discriminator
