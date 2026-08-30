"""CASIA input pipeline for DCGAN training at 64 x 64 in [-1, 1]."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset


class GANImageDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest_path: str | Path, image_size: int = 64) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = self.manifest_path.parents[2]
        self.image_size = image_size
        with self.manifest_path.open(newline="", encoding="utf-8") as file:
            self.records = list(csv.DictReader(file))
        if not self.records:
            raise ValueError(f"Empty split manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        configured = Path(record["image_path"])
        path = configured if configured.is_absolute() else self.project_root / configured
        try:
            with Image.open(path) as image:
                resized = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                array = np.array(resized, dtype=np.float32, copy=True) / 127.5 - 1.0
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Could not load GAN image: {path}") from exc
        return {
            "image": torch.from_numpy(array).permute(2, 0, 1).contiguous(),
            "label": int(record["label"]), "class_name": record["class_name"],
            "path": str(path),
        }


def create_gan_dataloaders(
    splits_dir: str | Path = "data/splits", image_size: int = 64,
    batch_size: int = 64, num_workers: int = 0, seed: int = 42,
) -> dict[str, DataLoader]:
    splits_dir = Path(splits_dir)
    generator = torch.Generator().manual_seed(seed)
    datasets = {
        name: GANImageDataset(splits_dir / f"{name}.csv", image_size)
        for name in ("train", "validation", "test")
    }
    return {
        name: DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train",
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=name == "train", persistent_workers=num_workers > 0,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
