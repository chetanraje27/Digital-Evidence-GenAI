"""Train the standard convolutional Autoencoder with MSE reconstruction loss."""

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
from autoencoder import ConvolutionalAutoencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[AESample],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Run one complete train or validation epoch and return sample-weighted MSE."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructions = model(images)
            loss = loss_function(reconstructions, images)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    return loss_sum / sample_count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    validation_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "config": {
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
        },
        path,
    )


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "validation_loss", "learning_rate"]
        )
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(history: list[dict[str, float | int]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    validation_losses = [float(row["validation_loss"]) for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", label="Training loss")
    axis.plot(epochs, validation_losses, marker="o", label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="MSE loss", title="Standard Autoencoder Training")
    axis.grid(alpha=0.3)
    axis.legend()
    axis.set_xticks(epochs)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_reconstruction_grid(
    model: nn.Module, loader: DataLoader[AESample], device: torch.device, path: Path
) -> None:
    model.eval()
    batch = next(iter(loader))
    inputs = batch["image"].to(device)
    with torch.no_grad():
        reconstructions = model(inputs)
    inputs = inputs.cpu()
    reconstructions = reconstructions.cpu()

    sample_count = min(5, inputs.shape[0])
    figure, axes = plt.subplots(2, sample_count, figsize=(15, 6), squeeze=False)
    for index in range(sample_count):
        axes[0][index].imshow(inputs[index].permute(1, 2, 0).numpy())
        axes[1][index].imshow(reconstructions[index].permute(1, 2, 0).numpy())
        axes[0][index].set_title("Original")
        axes[1][index].set_title("Reconstructed")
        axes[0][index].axis("off")
        axes[1][index].axis("off")
    figure.suptitle("Best Standard Autoencoder — Sanity Reconstruction")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs_to_run = 1 if args.smoke_test else args.max_epochs
    loaders = create_ae_dataloaders(
        splits_dir=args.splits_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = ConvolutionalAutoencoder().to(device)
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    start_time = time.perf_counter()

    for epoch in range(1, epochs_to_run + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(model, loaders["train"], loss_function, device, optimizer)
        validation_loss = run_epoch(model, loaders["validation"], loss_function, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.checkpoint_path, model, optimizer, epoch, validation_loss, args
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{epochs_to_run:02d} | train={train_loss:.8f} | "
            f"validation={validation_loss:.8f} | seconds={time.perf_counter() - epoch_start:.1f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            early_stopping_triggered = True
            break

    training_time = time.perf_counter() - start_time
    save_history(history, args.history_path)
    save_loss_curve(history, args.curve_path)

    # Reload the best validation checkpoint before the sanity reconstruction.
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_reconstruction_grid(model, loaders["validation"], device, args.grid_path)

    return {
        "device": str(device),
        "epochs_completed": len(history),
        "first_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "early_stopping_triggered": early_stopping_triggered,
        "checkpoint_path": str(args.checkpoint_path),
        "training_time_seconds": training_time,
        "smoke_test": args.smoke_test,
        "warning": (
            "CPU smoke test only; run without --smoke-test on CUDA/Google Colab for full training."
            if args.smoke_test and device.type == "cpu"
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_autoencoder.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/ae_training_history.csv"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/ae/ae_training_curve.png"))
    parser.add_argument(
        "--grid-path", type=Path, default=Path("outputs/ae/trained_reconstruction_grid.png")
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
