"""Reusable generation wrapper for the trained DCGAN checkpoints."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import torch

from dcgan import DCGANDiscriminator, DCGANGenerator


def _load_colab_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    # Python 3.13 pickled pathlib._local.PosixPath in the configuration. Map that
    # harmless path metadata to PurePosixPath while retaining weights_only=True.
    with torch.serialization.safe_globals([(PurePosixPath, "pathlib._local.PosixPath")]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported GAN checkpoint format: {path}")
    return checkpoint


class GANInference:
    def __init__(
        self, generator_path: str | Path, discriminator_path: str | Path,
        device: torch.device, latent_dim: int = 100,
    ) -> None:
        generator_path, discriminator_path = Path(generator_path), Path(discriminator_path)
        for name, path in (("generator", generator_path), ("discriminator", discriminator_path)):
            if not path.is_file(): raise FileNotFoundError(f"GAN {name} checkpoint not found: {path}")
        self.device, self.latent_dim = device, latent_dim
        self.generator = DCGANGenerator(latent_dim).to(device)
        self.discriminator = DCGANDiscriminator().to(device)
        self.generator.load_state_dict(_load_colab_checkpoint(generator_path, device)["model_state_dict"])
        self.discriminator.load_state_dict(_load_colab_checkpoint(discriminator_path, device)["model_state_dict"])
        self.generator.eval(); self.discriminator.eval()

    def generate(self, count: int = 1) -> list[np.ndarray]:
        if not 1 <= count <= 8: raise ValueError("GAN sample count must be between 1 and 8")
        with torch.no_grad():
            noise = torch.randn(count, self.latent_dim, 1, 1, device=self.device)
            generated = ((self.generator(noise) + 1.0) / 2.0).clamp(0, 1)
        return [image.permute(1, 2, 0).cpu().numpy() for image in generated]
