"""Reusable inference wrapper for Transformer Forensic Autoencoder V2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ae_inference import image_metrics, preprocess_image
from checkpoint_utils import validate_checkpoint_file
from transformer import TransformerForensicAutoencoder


class TransformerInference:
    def __init__(self, checkpoint_path: str | Path, device: torch.device) -> None:
        path = validate_checkpoint_file(checkpoint_path, "Vision Transformer V2")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        required = ("image_size", "patch_size", "embed_dim", "num_heads", "encoder_layers", "decoder_layers", "ff_dim")
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise ValueError(f"Transformer checkpoint metadata is missing: {missing}")
        self.device = device
        self.metadata = {key: checkpoint.get(key) for key in checkpoint if key != "model_state_dict"}
        self.model = TransformerForensicAutoencoder(
            image_size=int(checkpoint["image_size"]),
            patch_size=int(checkpoint["patch_size"]),
            embed_dim=int(checkpoint["embed_dim"]),
            num_heads=int(checkpoint["num_heads"]),
            encoder_layers=int(checkpoint["encoder_layers"]),
            decoder_layers=int(checkpoint["decoder_layers"]),
            ff_dim=int(checkpoint["ff_dim"]),
        ).to(device)
        incompatible = self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Transformer checkpoint mismatch: {incompatible}")
        self.model.eval()

    def reconstruct(self, image: Image.Image):
        original, tensor = preprocess_image(image, int(self.metadata["image_size"]))
        tensor = tensor.to(self.device)
        with torch.no_grad():
            reconstructed, latent, attention_maps = self.model(tensor)
        metrics = image_metrics(tensor, reconstructed)
        threshold = float(self.metadata.get("anomaly_threshold", float("nan")))
        metrics["reconstruction_error"] = metrics["mse"]
        metrics["anomaly_threshold"] = threshold
        output = reconstructed[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        attention = attention_maps[-1][0].mean(dim=0).cpu().numpy()
        side = int(round(attention.size ** 0.5))
        attention = attention.reshape(side, side)
        attention = (attention - attention.min()) / (np.ptp(attention) + 1e-8)
        return original, output, metrics, latent[0].cpu().numpy(), attention
