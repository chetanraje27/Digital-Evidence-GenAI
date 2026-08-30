"""Fine-tune the standard Autoencoder for Gaussian image denoising."""

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

from ae_dataset import AutoencoderImageDataset
from autoencoder import ConvolutionalAutoencoder
from denoising_dataset import DenoisingImageDataset, DenoisingSample


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_denoising_loaders(args: argparse.Namespace) -> dict[str, DataLoader[DenoisingSample]]:
    datasets = {
        name: DenoisingImageDataset(
            AutoencoderImageDataset(args.splits_dir / f"{name}.csv", args.image_size),
            noise_std=args.noise_std,
            seed=args.seed + offset,
        )
        for name, offset in (("train", 0), ("validation", 10_000), ("test", 20_000))
    }
    generator = torch.Generator().manual_seed(args.seed)
    return {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader[DenoisingSample],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            denoised = model(noisy)
            loss = loss_function(denoised, clean)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * clean.shape[0]
        total_images += clean.shape[0]
    return total_loss / total_images


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "validation_loss", "learning_rate"]
        )
        writer.writeheader()
        writer.writerows(history)


def save_curve(history: list[dict[str, float | int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Training")
    axis.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axis.set(title="Denoising Autoencoder Training", xlabel="Epoch", ylabel="MSE loss")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.3)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_denoising_loaders(args)
    model = ConvolutionalAutoencoder().to(device)
    source = torch.load(args.standard_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(source["model_state_dict"])
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopping = False
    start = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        train_dataset = loaders["train"].dataset
        assert isinstance(train_dataset, DenoisingImageDataset)
        train_dataset.set_epoch(epoch)
        train_loss = run_epoch(model, loaders["train"], loss_function, device, optimizer)
        validation_loss = run_epoch(model, loaders["validation"], loss_function, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "noise_std": args.noise_std,
                "initialized_from": str(args.standard_checkpoint),
                "config": {
                    "image_size": args.image_size,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                },
            }, args.checkpoint_path)
        else:
            stale_epochs += 1
        print(
            f"Epoch {epoch:02d}/{args.max_epochs:02d} | train={train_loss:.8f} | "
            f"validation={validation_loss:.8f} | seconds={time.perf_counter()-epoch_start:.1f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            early_stopping = True
            break

    training_time = time.perf_counter() - start
    save_history(history, args.history_path)
    save_curve(history, args.curve_path)
    best = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    summary: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "noise_std": args.noise_std,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "early_stopping_triggered": early_stopping,
        "training_time_seconds": training_time,
        "checkpoint_path": str(args.checkpoint_path),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--standard-checkpoint", type=Path, default=Path("checkpoints/best_autoencoder.pth"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_denoising_autoencoder.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/dae_training_history.csv"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/ae/dae_training_curve.png"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/dae_training_summary.json"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
