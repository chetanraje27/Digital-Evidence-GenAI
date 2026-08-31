"""Reusable reconstruction and generation wrapper for trained VAE V2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ae_inference import _state_dict, image_metrics, preprocess_image
from vae import ConvolutionalVAE


class VAEInference:
    def __init__(self, checkpoint_path: str | Path, device: torch.device, latent_dim: int = 128) -> None:
        path = Path(checkpoint_path)
        if not path.is_file(): raise FileNotFoundError(f"VAE checkpoint not found: {path}")
        self.device, self.latent_dim = device, latent_dim
        self.model = ConvolutionalVAE(latent_dim).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(_state_dict(checkpoint)); self.model.eval()

    def reconstruct(self, image: Image.Image) -> tuple[Image.Image, np.ndarray, dict[str, float]]:
        original_image, tensor = preprocess_image(image, 128)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(tensor)
            reconstructed = self.model.decode(mu)
        metrics = image_metrics(tensor, reconstructed)
        output = reconstructed.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return original_image, output, metrics

    def generate(self) -> np.ndarray:
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim, device=self.device)
            generated = self.model.decode(z)
        return generated.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
