"""Run one untrained forward pass to validate the AE architecture."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from autoencoder import ConvolutionalAutoencoder


SEED = 42
INPUT_SHAPE = (32, 3, 128, 128)
EXPECTED_LATENT_SHAPE = (32, 32, 8, 8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_untrained_grid(
    inputs: torch.Tensor, reconstructions: torch.Tensor, output_path: Path
) -> None:
    sample_count = min(5, inputs.shape[0])
    figure, axes = plt.subplots(2, sample_count, figsize=(15, 6), squeeze=False)
    for index in range(sample_count):
        axes[0][index].imshow(inputs[index].permute(1, 2, 0).numpy())
        axes[0][index].set_title("Original")
        axes[1][index].imshow(reconstructions[index].permute(1, 2, 0).numpy())
        axes[1][index].set_title("Untrained reconstruction")
        axes[0][index].axis("off")
        axes[1][index].axis("off")
    figure.suptitle("UNTRAINED MODEL — Debugging Only")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> dict[str, object]:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_ae_dataloaders(
        splits_dir="data/splits", image_size=128, batch_size=32, num_workers=0, seed=SEED
    )
    inputs = next(iter(loaders["train"]))["image"].to(device)
    model = ConvolutionalAutoencoder().to(device)
    model.eval()

    loss_function = nn.MSELoss()
    with torch.no_grad():
        latent = model.encode(inputs)
        reconstructions = model.decode(latent)
        forward_reconstructions = model(inputs)
        initial_mse = float(loss_function(reconstructions, inputs).item())

    input_values = int(inputs[0].numel())
    latent_values = int(latent[0].numel())
    compression_ratio = input_values / latent_values
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    output_min = float(reconstructions.min().item())
    output_max = float(reconstructions.max().item())
    nan_present = bool(torch.isnan(reconstructions).any().item())

    assert tuple(inputs.shape) == INPUT_SHAPE
    assert tuple(latent.shape) == EXPECTED_LATENT_SHAPE
    assert reconstructions.shape == inputs.shape
    assert torch.equal(forward_reconstructions, reconstructions)
    assert latent_values < input_values
    assert compression_ratio > 1.0
    assert not nan_present
    assert 0.0 <= output_min <= output_max <= 1.0

    summary: dict[str, object] = {
        "input_shape": list(inputs.shape),
        "latent_shape": list(latent.shape),
        "output_shape": list(reconstructions.shape),
        "input_values_per_image": input_values,
        "latent_values_per_image": latent_values,
        "compression_ratio": compression_ratio,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "initial_untrained_mse": initial_mse,
        "reconstruction_output_min": output_min,
        "reconstruction_output_max": output_max,
        "nan_present": nan_present,
        "device": str(device),
    }
    results_path = Path("results/ae_architecture_summary.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_untrained_grid(
        inputs.cpu(), reconstructions.cpu(), Path("outputs/ae/untrained_reconstruction_grid.png")
    )
    return summary


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
    print(json.dumps(main(), indent=2))
