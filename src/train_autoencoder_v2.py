"""Quality-focused training/fine-tuning for the existing convolutional Autoencoder.

The architecture, CASIA splits, preprocessing, and MSE objective are unchanged.
V2 artifacts use separate paths so the validated baseline is never overwritten.
"""

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
from torch.optim.lr_scheduler import ReduceLROnPlateau

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from autoencoder import ConvolutionalAutoencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_state(path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, {}


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    amp_enabled = device.type == "cuda"

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                reconstructions = model(images)
                loss = criterion(reconstructions, images)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        count = images.shape[0]
        loss_sum += float(loss.item()) * count
        sample_count += count
    if sample_count == 0:
        raise RuntimeError("No samples were processed.")
    return loss_sum / sample_count


def save_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "epoch", "train_loss", "validation_loss", "learning_rate", "epoch_time_seconds"
        ])
        writer.writeheader()
        writer.writerows(history)


def save_curve(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train MSE")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation MSE")
    axes[0].set(title="AE V2 reconstruction loss", xlabel="Fine-tuning epoch", ylabel="MSE")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["learning_rate"] for row in history], color="#7c3aed")
    axes[1].set(title="Learning-rate schedule", xlabel="Fine-tuning epoch", ylabel="Learning rate")
    axes[1].set_yscale("log"); axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_grid(model, loader, device, path: Path, count: int = 6) -> None:
    batch = next(iter(loader))
    images = batch["image"][:count].to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(images).clamp(0, 1)
    images, outputs = images.cpu(), outputs.cpu()
    figure, axes = plt.subplots(2, len(images), figsize=(3 * len(images), 6), squeeze=False)
    for index in range(len(images)):
        axes[0, index].imshow(images[index].permute(1, 2, 0)); axes[0, index].set_title("Original")
        axes[1, index].imshow(outputs[index].permute(1, 2, 0)); axes[1, index].set_title("Reconstructed")
        axes[0, index].axis("off"); axes[1, index].axis("off")
    figure.suptitle("Best AE V2 validation reconstructions")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("Full AE V2 training requires a CUDA runtime. Enable a Colab GPU.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loaders = create_ae_dataloaders(args.splits_dir, args.image_size, args.batch_size,
                                    args.num_workers, args.seed)
    model = ConvolutionalAutoencoder().to(device)
    initialized_from = None
    baseline_validation_loss = None
    if args.initial_checkpoint and Path(args.initial_checkpoint).is_file():
        state, metadata = _load_model_state(Path(args.initial_checkpoint), device)
        model.load_state_dict(state, strict=True)
        initialized_from = str(args.initial_checkpoint)
        baseline_validation_loss = metadata.get("validation_loss")

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                  patience=args.lr_patience, min_lr=args.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.MSELoss()
    max_batches = args.smoke_batches if args.smoke_test else None
    epochs_to_run = 1 if args.smoke_test else args.max_epochs
    best_loss = float(baseline_validation_loss) if baseline_validation_loss is not None else float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopping = False
    history = []
    start = time.perf_counter()

    # Ensure a valid V2 checkpoint exists even if fine-tuning never beats the baseline.
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0,
                "validation_loss": best_loss, "initialized_from": initialized_from}, args.checkpoint_path)

    for epoch in range(1, epochs_to_run + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, max_batches)
        validation_loss = run_epoch(model, loaders["validation"], criterion, device,
                                    max_batches=max_batches)
        scheduler.step(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "learning_rate": learning_rate,
                        "epoch_time_seconds": elapsed})

        if validation_loss < best_loss - args.min_delta:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
                "validation_loss": validation_loss, "initialized_from": initialized_from,
                "config": vars(args),
            }, args.checkpoint_path)
        else:
            stale_epochs += 1
        print(f"Epoch {epoch:03d}/{epochs_to_run:03d} | train={train_loss:.8f} | "
              f"validation={validation_loss:.8f} | lr={learning_rate:.2e} | seconds={elapsed:.1f}", flush=True)
        if not args.smoke_test and stale_epochs >= args.patience:
            early_stopping = True
            break

    total_time = time.perf_counter() - start
    save_history(history, args.history_path)
    save_curve(history, args.curve_path)
    state, checkpoint = _load_model_state(args.checkpoint_path, device)
    model.load_state_dict(state, strict=True)
    save_grid(model, loaders["validation"], device, args.grid_path)
    summary = {
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs_completed": len(history), "first_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"], "best_validation_loss": best_loss,
        "best_epoch": best_epoch, "early_stopping_triggered": early_stopping,
        "training_time_seconds": total_time, "checkpoint_path": str(args.checkpoint_path),
        "initialized_from": initialized_from, "baseline_validation_loss": baseline_validation_loss,
        "smoke_test": args.smoke_test,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("checkpoints/best_autoencoder.pth"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_autoencoder_v2.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/ae_v2_training_history.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/ae_v2_training_summary.json"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/ae_v2/training_curve.png"))
    parser.add_argument("--grid-path", type=Path, default=Path("outputs/ae_v2/reconstruction_grid.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
