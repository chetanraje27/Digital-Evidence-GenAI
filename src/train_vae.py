"""Train the convolutional VAE on the existing CASIA split manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import AESample, create_ae_dataloaders
from vae import ConvolutionalVAE


HISTORY_FIELDS = [
    "epoch",
    "train_total_loss",
    "train_reconstruction_loss",
    "train_kl_loss",
    "validation_total_loss",
    "validation_reconstruction_loss",
    "validation_kl_loss",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vae_losses(
    reconstructions: torch.Tensor,
    inputs: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, pixel-mean MSE, and mean per-sample latent KL loss."""
    reconstruction_loss = nn.functional.mse_loss(reconstructions, inputs)
    kl_loss = (-0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )).mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def run_epoch(
    model: ConvolutionalVAE,
    loader: DataLoader[AESample],
    device: torch.device,
    beta: float,
    optimizer: Adam | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    image_count = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstructions, mu, logvar = model(inputs)
            total, reconstruction, kl = vae_losses(
                reconstructions, inputs, mu, logvar, beta
            )
            if training:
                total.backward()
                optimizer.step()

        batch_size = inputs.shape[0]
        sums["total"] += total.item() * batch_size
        sums["reconstruction"] += reconstruction.item() * batch_size
        sums["kl"] += kl.item() * batch_size
        image_count += batch_size

    if image_count == 0:
        raise RuntimeError("No images were processed in the epoch")
    return {name: value / image_count for name, value in sums.items()}


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def save_curves(
    history: list[dict[str, float | int]], total_path: Path, component_path: Path
) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_total_loss"] for row in history], marker="o", label="Train")
    axis.plot(epochs, [row["validation_total_loss"] for row in history], marker="o", label="Validation")
    axis.set(title="VAE Total Loss", xlabel="Epoch", ylabel="Reconstruction + beta × KL")
    axis.grid(alpha=0.3)
    axis.legend()
    total_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(total_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [row["train_reconstruction_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_reconstruction_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Reconstruction Loss", xlabel="Epoch", ylabel="MSE")
    axes[1].plot(epochs, [row["train_kl_loss"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["validation_kl_loss"] for row in history], marker="o", label="Validation")
    axes[1].set(title="KL Divergence", xlabel="Epoch", ylabel="Mean KL per sample")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    component_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(component_path, dpi=160)
    plt.close(figure)


def save_reconstruction_grid(
    model: ConvolutionalVAE,
    loader: DataLoader[AESample],
    device: torch.device,
    path: Path,
) -> None:
    model.eval()
    inputs = next(iter(loader))["image"][:6].to(device)
    with torch.no_grad():
        mu, _ = model.encode(inputs)
        reconstructions = model.decode(mu)
    inputs, reconstructions = inputs.cpu(), reconstructions.cpu()
    figure, axes = plt.subplots(2, len(inputs), figsize=(16, 6), squeeze=False)
    for index in range(len(inputs)):
        axes[0][index].imshow(inputs[index].permute(1, 2, 0).numpy())
        axes[1][index].imshow(reconstructions[index].permute(1, 2, 0).numpy())
        axes[0][index].set_title("Original")
        axes[1][index].set_title("Reconstructed")
        axes[0][index].axis("off")
        axes[1][index].axis("off")
    figure.suptitle("Best VAE — Validation Reconstructions (using z = mu)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_random_samples(
    model: ConvolutionalVAE,
    latent_dim: int,
    device: torch.device,
    path: Path,
    sample_count: int = 16,
) -> None:
    model.eval()
    with torch.no_grad():
        samples = model.decode(torch.randn(sample_count, latent_dim, device=device)).cpu()
    columns = 4
    rows = (sample_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, 10), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < sample_count:
            axis.imshow(samples[index].permute(1, 2, 0).numpy())
            axis.set_title(f"Sample {index + 1}")
        axis.axis("off")
    figure.suptitle("VAE Random Samples: z ~ N(0, I)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    model = ConvolutionalVAE(args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    epochs_to_run = 1 if args.smoke_test else args.max_epochs
    max_batches = args.smoke_batches if args.smoke_test else None

    history: list[dict[str, float | int]] = []
    best_validation_total = float("inf")
    best_epoch = 0
    best_validation_components = {"reconstruction": float("inf"), "kl": float("inf")}
    stale_epochs = 0
    early_stopping = False
    start = time.perf_counter()

    for epoch in range(1, epochs_to_run + 1):
        epoch_start = time.perf_counter()
        training = run_epoch(model, loaders["train"], device, args.beta, optimizer, max_batches)
        validation = run_epoch(model, loaders["validation"], device, args.beta, None, max_batches)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": training["total"],
            "train_reconstruction_loss": training["reconstruction"],
            "train_kl_loss": training["kl"],
            "validation_total_loss": validation["total"],
            "validation_reconstruction_loss": validation["reconstruction"],
            "validation_kl_loss": validation["kl"],
        }
        history.append(row)

        if validation["total"] < best_validation_total:
            best_validation_total = validation["total"]
            best_validation_components = {
                "reconstruction": validation["reconstruction"], "kl": validation["kl"]
            }
            best_epoch = epoch
            stale_epochs = 0
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_total_loss": validation["total"],
                "validation_reconstruction_loss": validation["reconstruction"],
                "validation_kl_loss": validation["kl"],
                "config": {
                    "latent_dim": args.latent_dim, "beta": args.beta,
                    "image_size": args.image_size, "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate, "seed": args.seed,
                },
            }, args.checkpoint_path)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:02d}/{epochs_to_run:02d} | "
            f"train_total={training['total']:.8f} | val_total={validation['total']:.8f} | "
            f"recon={validation['reconstruction']:.8f} | kl={validation['kl']:.6f} | "
            f"seconds={time.perf_counter()-epoch_start:.1f}", flush=True,
        )
        if stale_epochs >= args.patience:
            early_stopping = True
            break

    training_time = time.perf_counter() - start
    save_history(history, args.history_path)
    save_curves(history, args.total_curve_path, args.component_curve_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_reconstruction_grid(model, loaders["validation"], device, args.reconstruction_grid_path)
    save_random_samples(model, args.latent_dim, device, args.random_samples_path)

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "first_train_total_loss": history[0]["train_total_loss"],
        "final_train_total_loss": history[-1]["train_total_loss"],
        "best_validation_total_loss": best_validation_total,
        "best_validation_reconstruction_loss": best_validation_components["reconstruction"],
        "best_validation_kl_loss": best_validation_components["kl"],
        "early_stopping_triggered": early_stopping,
        "training_time_seconds": training_time,
        "checkpoint_path": str(args.checkpoint_path),
        "smoke_test": args.smoke_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_vae.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/vae_training_history.csv"))
    parser.add_argument("--total-curve-path", type=Path, default=Path("outputs/vae/vae_total_loss_curve.png"))
    parser.add_argument("--component-curve-path", type=Path, default=Path("outputs/vae/vae_reconstruction_kl_curve.png"))
    parser.add_argument("--reconstruction-grid-path", type=Path, default=Path("outputs/vae/vae_reconstruction_grid.png"))
    parser.add_argument("--random-samples-path", type=Path, default=Path("outputs/vae/vae_random_samples.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
