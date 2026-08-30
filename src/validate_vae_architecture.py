"""Run a one-batch, untrained forward-pass smoke test for the VAE."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ae_dataset import create_ae_dataloaders
from vae import ConvolutionalVAE


SEED = 42
LATENT_DIM = 128
EXPECTED_INPUT_SHAPE = (32, 3, 128, 128)
EXPECTED_FEATURE_SHAPE = (32, 128, 8, 8)
EXPECTED_LATENT_SHAPE = (32, LATENT_DIM)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence per sample, summed across latent dimensions."""
    per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )
    return per_sample.mean()


def validate() -> dict[str, object]:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_ae_dataloaders(
        splits_dir="data/splits",
        image_size=128,
        batch_size=32,
        num_workers=0,
        seed=SEED,
    )["train"]
    inputs = next(iter(train_loader))["image"].to(device)
    model = ConvolutionalVAE(latent_dim=LATENT_DIM).to(device)
    model.eval()

    reconstruction_loss_function = nn.MSELoss()
    with torch.no_grad():
        encoded_features = model.encode_features(inputs)
        mu, logvar = model.encode(inputs)
        z = model.reparameterize(mu, logvar)
        reconstructions = model.decode(z)
        forward_reconstruction, forward_mu, forward_logvar = model(inputs)
        reconstruction_loss = reconstruction_loss_function(reconstructions, inputs)
        initial_kl_loss = kl_divergence(mu, logvar)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    output_min = float(reconstructions.min().item())
    output_max = float(reconstructions.max().item())
    nan_present = bool(
        torch.isnan(reconstructions).any()
        or torch.isnan(mu).any()
        or torch.isnan(logvar).any()
        or torch.isnan(z).any()
    )

    assert tuple(inputs.shape) == EXPECTED_INPUT_SHAPE
    assert tuple(encoded_features.shape) == EXPECTED_FEATURE_SHAPE
    assert tuple(mu.shape) == EXPECTED_LATENT_SHAPE
    assert tuple(logvar.shape) == EXPECTED_LATENT_SHAPE
    assert tuple(z.shape) == EXPECTED_LATENT_SHAPE
    assert reconstructions.shape == inputs.shape
    assert forward_reconstruction.shape == inputs.shape
    assert tuple(forward_mu.shape) == EXPECTED_LATENT_SHAPE
    assert tuple(forward_logvar.shape) == EXPECTED_LATENT_SHAPE
    assert model.latent_dim == LATENT_DIM
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
    assert torch.isfinite(reconstruction_loss)
    assert torch.isfinite(initial_kl_loss)
    assert not nan_present
    assert 0.0 <= output_min <= output_max <= 1.0

    summary: dict[str, object] = {
        "input_shape": list(inputs.shape),
        "encoded_feature_shape": list(encoded_features.shape),
        "mu_shape": list(mu.shape),
        "logvar_shape": list(logvar.shape),
        "latent_shape": list(z.shape),
        "output_shape": list(reconstructions.shape),
        "latent_dim": model.latent_dim,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "initial_reconstruction_loss": float(reconstruction_loss.item()),
        "initial_kl_loss": float(initial_kl_loss.item()),
        "output_min": output_min,
        "output_max": output_max,
        "nan_present": nan_present,
        "device": str(device),
    }
    summary_path = Path("results/vae_architecture_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
