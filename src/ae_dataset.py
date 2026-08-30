"""PyTorch input pipeline for the standard CASIA Autoencoder experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset


class AESample(TypedDict):
    image: torch.Tensor
    label: int
    path: str
    class_name: str


class AutoencoderImageDataset(Dataset[AESample]):
    """Load RGB images listed in a split CSV without ImageNet normalization."""

    def __init__(self, manifest_path: str | Path, image_size: int = 128) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = self.manifest_path.parents[2]
        self.image_size = image_size

        with self.manifest_path.open(newline="", encoding="utf-8") as file:
            self.records = list(csv.DictReader(file))
        if not self.records:
            raise ValueError(f"Split manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AESample:
        record = self.records[index]
        configured_path = Path(record["image_path"])
        image_path = configured_path if configured_path.is_absolute() else self.project_root / configured_path

        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                resized = rgb_image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                # Copy creates writable memory owned by the tensor.
                array = np.array(resized, dtype=np.float32, copy=True) / 255.0
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Could not load image: {image_path}") from exc

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return {
            "image": tensor,
            "label": int(record["label"]),
            "path": str(image_path),
            "class_name": record["class_name"],
        }


def create_ae_dataloaders(
    splits_dir: str | Path = "data/splits",
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader[AESample]]:
    """Create reproducible loaders; num_workers=0 is the reliable Windows default."""
    splits_dir = Path(splits_dir)
    datasets = {
        "train": AutoencoderImageDataset(splits_dir / "train.csv", image_size),
        "validation": AutoencoderImageDataset(splits_dir / "validation.csv", image_size),
        "test": AutoencoderImageDataset(splits_dir / "test.csv", image_size),
    }
    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
