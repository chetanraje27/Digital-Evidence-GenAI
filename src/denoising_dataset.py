"""Reusable noisy/clean CASIA pairs for the Denoising Autoencoder."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch.utils.data import Dataset

from ae_dataset import AESample, AutoencoderImageDataset


class DenoisingSample(TypedDict):
    noisy: torch.Tensor
    clean: torch.Tensor
    label: int
    path: str
    class_name: str


def add_gaussian_noise(
    clean: torch.Tensor, noise_std: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Add zero-mean Gaussian noise and retain the valid [0,1] pixel range."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    noise = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    return torch.clamp(clean + noise * noise_std, 0.0, 1.0)


class DenoisingImageDataset(Dataset[DenoisingSample]):
    """Wrap the existing AE dataset and create reproducible noisy inputs."""

    def __init__(
        self,
        clean_dataset: AutoencoderImageDataset,
        noise_std: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.noise_std = noise_std
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change training noise deterministically between epochs."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, index: int) -> DenoisingSample:
        sample: AESample = self.clean_dataset[index]
        generator = torch.Generator().manual_seed(
            self.seed + index + self.epoch * len(self.clean_dataset)
        )
        clean = sample["image"]
        noisy = add_gaussian_noise(clean, self.noise_std, generator)
        return {
            "noisy": noisy,
            "clean": clean,
            "label": sample["label"],
            "path": sample["path"],
            "class_name": sample["class_name"],
        }
