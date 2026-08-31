"""Reusable inference wrapper for the trained convolutional Autoencoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from autoencoder import ConvolutionalAutoencoder


def _state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def preprocess_image(image: Image.Image, size: int = 128) -> tuple[Image.Image, torch.Tensor]:
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
    return rgb, tensor


def image_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    first = original.squeeze(0).detach().cpu()
    second = reconstructed.squeeze(0).detach().cpu()
    mse = float(torch.mean((first - second) ** 2))
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    ssim = float(structural_similarity(
        first.permute(1, 2, 0).numpy(), second.permute(1, 2, 0).numpy(),
        channel_axis=2, data_range=1.0,
    ))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


class AutoencoderInference:
    def __init__(self, checkpoint_path: str | Path, device: torch.device) -> None:
        path = Path(checkpoint_path)
        if not path.is_file(): raise FileNotFoundError(f"AE checkpoint not found: {path}")
        self.device = device
        self.model = ConvolutionalAutoencoder().to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(_state_dict(checkpoint)); self.model.eval()

    def reconstruct(self, image: Image.Image) -> tuple[Image.Image, np.ndarray, dict[str, float]]:
        original_image, tensor = preprocess_image(image, 128)
        tensor = tensor.to(self.device)
        with torch.no_grad(): reconstructed = self.model(tensor)
        metrics = image_metrics(tensor, reconstructed)
        output = reconstructed.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return original_image, output, metrics
