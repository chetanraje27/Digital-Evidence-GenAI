"""Reusable inference wrapper for the current final VAE V5 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ae_inference import _state_dict, image_metrics, preprocess_image
from checkpoint_utils import validate_checkpoint_file
from vae import DEFAULT_LATENT_DIM, VAEV5


class VAEInference:
    """Load VAE V5 once and expose deterministic reconstruction and sampling."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ) -> None:
        path = validate_checkpoint_file(checkpoint_path, "VAE V5")
        self.device = device
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            latent_dim = int(checkpoint.get("latent_dim", latent_dim))
            self.checkpoint_metadata = {
                key: checkpoint.get(key)
                for key in ("epoch", "val_mse", "val_psnr", "val_kl", "beta", "l1_weight")
            }
        else:
            self.checkpoint_metadata = {}
        self.latent_dim = latent_dim
        self.model = VAEV5(latent_dim).to(device)
        incompatible = self.model.load_state_dict(_state_dict(checkpoint), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"VAE V5 checkpoint mismatch: {incompatible}")
        self.model.eval()

    def reconstruct(
        self, image: Image.Image
    ) -> tuple[Image.Image, np.ndarray, dict[str, float]]:
        original_image, tensor = preprocess_image(image, 128)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            reconstructed, _, _ = self.model.reconstruct(tensor, deterministic=True)
        metrics = image_metrics(tensor, reconstructed)
        metrics["reconstruction_error"] = metrics["mse"]
        output = reconstructed.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return original_image, output, metrics

    def reconstruct_variations(
        self,
        image: Image.Image,
        n_samples: int = 3,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> tuple[Image.Image, np.ndarray, list[dict[str, Any]], dict[str, float]]:
        """Return deterministic and image-conditioned stochastic reconstructions.

        Each stochastic latent is sampled from the uploaded image's learned
        posterior using ``mu + temperature * std * epsilon``. The image's own
        encoder skip tensors are reused for every decode.
        """
        if not 1 <= n_samples <= 8:
            raise ValueError("n_samples must be between 1 and 8")
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be a finite non-negative number")

        original_image, tensor = preprocess_image(image, 128)
        tensor = tensor.to(self.device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        with torch.no_grad():
            mu, logvar, e1, e2, e3, e4 = self.model.encode(tensor)
            std = torch.exp(0.5 * logvar)
            deterministic = self.model.decode(mu, e1, e2, e3, e4)
            deterministic_metrics = image_metrics(tensor, deterministic)
            deterministic_metrics["reconstruction_error"] = deterministic_metrics["mse"]

            variations: list[dict[str, Any]] = []
            for _ in range(n_samples):
                epsilon = torch.randn(
                    std.shape,
                    generator=generator,
                    device=self.device,
                    dtype=std.dtype,
                )
                z = mu + temperature * std * epsilon
                sampled = self.model.decode(z, e1, e2, e3, e4)
                sample_metrics = image_metrics(tensor, sampled)
                variations.append(
                    {
                        "image": self._to_array(sampled),
                        "metrics": sample_metrics,
                        "latent_l2_from_mu": float(torch.linalg.vector_norm(z - mu)),
                        "mean_abs_delta_from_deterministic": float(
                            torch.mean(torch.abs(sampled - deterministic))
                        ),
                        "rms_delta_from_deterministic": float(
                            torch.sqrt(torch.mean((sampled - deterministic) ** 2))
                        ),
                    }
                )

        return (
            original_image,
            self._to_array(deterministic),
            variations,
            deterministic_metrics,
        )

    @staticmethod
    def _to_array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    def generate(self) -> np.ndarray:
        """Decode a prior sample with zero skips; output is synthetic research data."""
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim, device=self.device)
            generated = self.model.decode(z)
        return generated.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
