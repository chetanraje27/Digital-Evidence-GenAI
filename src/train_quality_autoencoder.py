"""Train the residual quality Autoencoder on the canonical CASIA splits."""

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
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from quality_autoencoder import ARCHITECTURE_NAME, COMPRESSION_RATIO, QualityAutoencoder
from train_autoencoder_v2 import checkpoint_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def differentiable_ssim(
    reconstruction: torch.Tensor, target: torch.Tensor, window_size: int = 7
) -> torch.Tensor:
    """Return mean local SSIM using a differentiable uniform window."""
    padding = window_size // 2
    mu_x = F.avg_pool2d(reconstruction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(reconstruction * reconstruction, window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(reconstruction * target, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()


def edge_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match horizontal and vertical finite-difference image gradients."""
    recon_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    recon_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy))


class QualityReconstructionLoss(nn.Module):
    """Pixel, structural-similarity, and edge-preservation objective."""

    def __init__(self, l1_weight: float, ssim_weight: float, edge_weight: float) -> None:
        super().__init__()
        if min(l1_weight, ssim_weight, edge_weight) < 0:
            raise ValueError("Loss weights must be non-negative.")
        if not np.isclose(l1_weight + ssim_weight + edge_weight, 1.0):
            raise ValueError("Loss weights must sum to 1.0.")
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

    def components(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        l1 = F.l1_loss(reconstruction, target)
        ssim = differentiable_ssim(reconstruction, target)
        edges = edge_loss(reconstruction, target)
        total = self.l1_weight * l1 + self.ssim_weight * (1.0 - ssim) + self.edge_weight * edges
        mse = F.mse_loss(reconstruction, target)
        return {"loss": total, "l1": l1, "ssim": ssim, "edge": edges, "mse": mse}

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(reconstruction, target)["loss"]


def run_epoch(
    model: nn.Module,
    loader,
    criterion: QualityReconstructionLoss,
    device: torch.device,
    optimizer=None,
    scaler=None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("loss", "l1", "ssim", "edge", "mse")}
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
            parts = criterion.components(reconstructions.float(), images.float())
            if training:
                scaler.scale(parts["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        count = images.shape[0]
        for name in totals:
            totals[name] += float(parts[name].detach().item()) * count
        sample_count += count
    if sample_count == 0:
        raise RuntimeError("No samples were processed.")
    return {name: value / sample_count for name, value in totals.items()}


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_curve(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train objective")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation objective")
    axes[0].set(title="Quality objective", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["validation_mse"] for row in history], color="#2563eb")
    axes[1].set(title="Validation pixel error", xlabel="Epoch", ylabel="MSE")
    axes[2].plot(epochs, [row["validation_ssim"] for row in history], color="#059669")
    axes[2].set(title="Validation structural similarity", xlabel="Epoch", ylabel="SSIM")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_grid(model: nn.Module, loader, device: torch.device, path: Path, count: int = 6) -> None:
    images = next(iter(loader))["image"][:count].to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(images).clamp(0, 1)
    images, outputs = images.cpu(), outputs.cpu()
    figure, axes = plt.subplots(2, len(images), figsize=(3 * len(images), 6), squeeze=False)
    for index in range(len(images)):
        axes[0, index].imshow(images[index].permute(1, 2, 0))
        axes[1, index].imshow(outputs[index].permute(1, 2, 0))
        axes[0, index].set_title("Original")
        axes[1, index].set_title("Reconstructed")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle("Quality Autoencoder validation reconstructions")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("Full quality-AE training requires CUDA. Enable a Colab GPU runtime.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    model = QualityAutoencoder().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = QualityReconstructionLoss(args.l1_weight, args.ssim_weight, args.edge_weight)
    max_batches = args.smoke_batches if args.smoke_test else None
    epochs_to_run = 1 if args.smoke_test else args.max_epochs
    best_ssim = -float("inf")
    best_validation_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    early_stopping = False
    start = time.perf_counter()
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs_to_run + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, max_batches)
        validation_metrics = run_epoch(
            model, loaders["validation"], criterion, device, max_batches=max_batches
        )
        scheduler.step(validation_metrics["ssim"])
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_ssim": train_metrics["ssim"],
            "validation_loss": validation_metrics["loss"],
            "validation_mse": validation_metrics["mse"],
            "validation_ssim": validation_metrics["ssim"],
            "validation_edge_loss": validation_metrics["edge"],
            "learning_rate": learning_rate,
            "epoch_time_seconds": elapsed,
        })

        improved = validation_metrics["ssim"] > best_ssim + args.min_delta
        if improved:
            best_ssim = validation_metrics["ssim"]
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "architecture": ARCHITECTURE_NAME,
                "compression_ratio": COMPRESSION_RATIO,
                "latent_shape": list(model.latent_shape),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_metrics["loss"],
                "validation_mse": validation_metrics["mse"],
                "validation_ssim": validation_metrics["ssim"],
                "selection_metric": "validation_ssim",
                "config": checkpoint_config(args),
            }, args.checkpoint_path)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d}/{epochs_to_run:03d} | "
            f"train_loss={train_metrics['loss']:.7f} | "
            f"val_mse={validation_metrics['mse']:.8f} | "
            f"val_ssim={validation_metrics['ssim']:.6f} | "
            f"lr={learning_rate:.2e} | seconds={elapsed:.1f}",
            flush=True,
        )
        if not args.smoke_test and stale_epochs >= args.patience:
            early_stopping = True
            break

    total_time = time.perf_counter() - start
    save_history(history, args.history_path)
    save_curve(history, args.curve_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    save_grid(model, loaders["validation"], device, args.grid_path)
    summary = {
        "architecture": ARCHITECTURE_NAME,
        "compression_ratio": COMPRESSION_RATIO,
        "latent_shape": list(model.latent_shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_ssim": best_ssim,
        "best_validation_mse": best_validation_mse,
        "selection_metric": "validation_ssim",
        "early_stopping_triggered": early_stopping,
        "training_time_seconds": total_time,
        "checkpoint_path": str(args.checkpoint_path),
        "smoke_test": args.smoke_test,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_quality_autoencoder_v1.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/quality_ae_v1_training_history.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/quality_ae_v1_training_summary.json"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/quality_ae_v1/training_curve.png"))
    parser.add_argument("--grid-path", type=Path, default=Path("outputs/quality_ae_v1/validation_reconstruction_grid.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--l1-weight", type=float, default=0.65)
    parser.add_argument("--ssim-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
