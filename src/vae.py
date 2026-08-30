"""Convolutional Variational Autoencoder for 128 x 128 RGB CASIA images."""

from __future__ import annotations

import torch
from torch import nn


class ConvolutionalVAE(nn.Module):
    """VAE with a 128-dimensional probabilistic latent representation."""

    encoded_channels = 128
    encoded_size = 8

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.flattened_feature_dim = (
            self.encoded_channels * self.encoded_size * self.encoded_size
        )

        self.encoder = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 128 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(self.flattened_feature_dim, latent_dim)
        self.logvar_head = nn.Linear(self.flattened_feature_dim, latent_dim)

        self.decoder_projection = nn.Linear(latent_dim, self.flattened_feature_dim)
        self.decoder = nn.Sequential(
            # 128 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 x 64 x 64 -> 3 x 128 x 128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final spatial encoder feature map for inspection."""
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map an input batch to latent distribution parameters."""
        features = self.encode_features(x)
        flattened = torch.flatten(features, start_dim=1)
        return self.mu_head(flattened), self.logvar_head(flattened)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z while retaining gradients through mu and log-variance."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Project a latent sample and reconstruct an RGB image."""
        projected = self.decoder_projection(z)
        feature_map = projected.view(
            z.shape[0], self.encoded_channels, self.encoded_size, self.encoded_size
        )
        return self.decoder(feature_map)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction and latent distribution parameters."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
