"""Train VAE V2 with linear KL warm-up while preserving all V1 artifacts."""

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
from torch.optim import Adam

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from train_vae import run_epoch
from vae import ConvolutionalVAE


FIELDS = [
    "epoch", "train_reconstruction_loss", "train_kl_loss", "train_total_loss",
    "validation_reconstruction_loss", "validation_kl_loss",
    "validation_total_loss", "beta", "epoch_time_seconds",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_for_epoch(epoch: int, target_beta: float, warmup_epochs: int) -> float:
    """Linear schedule: 10% of target at epoch 1, target at epoch 10."""
    if warmup_epochs <= 0:
        return target_beta
    return target_beta * min(epoch / warmup_epochs, 1.0)


def save_history(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_curves(rows: list[dict[str, float | int]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, key, title, ylabel in (
        (axes[0, 0], "total_loss", "Total Loss", "MSE + beta × KL"),
        (axes[0, 1], "reconstruction_loss", "Reconstruction Loss", "MSE"),
        (axes[1, 0], "kl_loss", "KL Divergence", "Mean KL per sample"),
    ):
        axis.plot(epochs, [row[f"train_{key}"] for row in rows], label="Train")
        axis.plot(epochs, [row[f"validation_{key}"] for row in rows], label="Validation")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=.3)
        axis.legend()
    axes[1, 1].plot(epochs, [row["beta"] for row in rows], color="tab:purple")
    axes[1, 1].axvline(10, linestyle="--", color="gray", label="Warm-up complete")
    axes[1, 1].set(title="KL Beta Schedule", xlabel="Epoch", ylabel="Beta")
    axes[1, 1].grid(alpha=.3)
    axes[1, 1].legend()
    figure.suptitle("VAE V2 Training — Linear KL Warm-up")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    model = ConvolutionalVAE(args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    epoch_limit = 1 if args.smoke_test else args.max_epochs
    batch_limit = args.smoke_batches if args.smoke_test else None
    rows: list[dict[str, float | int]] = []
    best_value = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopping = False
    started = time.perf_counter()

    for epoch in range(1, epoch_limit + 1):
        epoch_started = time.perf_counter()
        beta = beta_for_epoch(epoch, args.target_beta, args.warmup_epochs)
        training = run_epoch(model, loaders["train"], device, beta, optimizer, batch_limit)
        validation = run_epoch(model, loaders["validation"], device, beta, None, batch_limit)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_reconstruction_loss": training["reconstruction"],
            "train_kl_loss": training["kl"], "train_total_loss": training["total"],
            "validation_reconstruction_loss": validation["reconstruction"],
            "validation_kl_loss": validation["kl"],
            "validation_total_loss": validation["total"], "beta": beta,
            "epoch_time_seconds": time.perf_counter() - epoch_started,
        }
        rows.append(row)

        # Compare checkpoints only under the final objective. A lower warm-up beta
        # would otherwise make early total losses artificially incomparable.
        eligible = epoch >= args.warmup_epochs or args.smoke_test
        if eligible and validation["total"] < best_value:
            best_value = validation["total"]
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
                    "version": "vae_v2", "latent_dim": args.latent_dim,
                    "target_beta": args.target_beta, "warmup_epochs": args.warmup_epochs,
                    "image_size": args.image_size, "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate, "seed": args.seed,
                },
            }, args.checkpoint_path)
        elif eligible:
            stale_epochs += 1

        save_history(rows, args.history_path)
        print(
            f"Epoch {epoch:02d}/{epoch_limit:02d} | beta={beta:.7f} | "
            f"train_recon={training['reconstruction']:.8f} | train_kl={training['kl']:.6f} | "
            f"train_total={training['total']:.8f} | val_recon={validation['reconstruction']:.8f} | "
            f"val_kl={validation['kl']:.6f} | val_total={validation['total']:.8f} | "
            f"seconds={row['epoch_time_seconds']:.1f}", flush=True,
        )
        if eligible and stale_epochs >= args.patience:
            early_stopping = True
            break

    save_curves(rows, args.curve_path)
    summary = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs_completed": len(rows), "best_epoch": best_epoch,
        "best_validation_total_loss": best_value,
        "early_stopping_triggered": early_stopping,
        "training_time_seconds": time.perf_counter() - started,
        "checkpoint_path": str(args.checkpoint_path), "history_path": str(args.history_path),
        "target_beta": args.target_beta, "warmup_epochs": args.warmup_epochs,
        "smoke_test": args.smoke_test,
    }
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_vae_v2.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/vae_v2_training_history.csv"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/vae/vae_v2_loss_curves.png"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/vae_v2_training_summary.json"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-beta", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
