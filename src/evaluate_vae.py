"""Evaluate frozen VAE V5 reconstruction, anomaly behavior, and optional FID."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from checkpoint_utils import validate_checkpoint_file
from vae import DEFAULT_BETA_END, DEFAULT_LATENT_DIM, VAEV5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"count": len(rows)}
    for key in ("mse", "psnr", "ssim", "reconstruction_loss", "kl_loss", "total_loss"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def save_reconstruction_grid(samples: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(len(samples), 2, figsize=(8, 3 * len(samples)), squeeze=False)
    for row_index, sample in enumerate(samples):
        original = sample["original"]
        reconstructed = sample["reconstructed"]
        assert isinstance(original, torch.Tensor) and isinstance(reconstructed, torch.Tensor)
        label = str(sample["class_name"]).title()
        axes[row_index, 0].imshow(original.permute(1, 2, 0).numpy())
        axes[row_index, 1].imshow(reconstructed.permute(1, 2, 0).numpy())
        axes[row_index, 0].set_title(f"Original — {label}")
        axes[row_index, 1].set_title(f"VAE V5 reconstruction — {label}")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    figure.suptitle("Deterministic reconstruction (z = μ)", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_error_distribution(rows: list[dict[str, object]], path: Path) -> None:
    authentic = [float(row["mse"]) for row in rows if int(row["label"]) == 0]
    tampered = [float(row["mse"]) for row in rows if int(row["label"]) == 1]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(authentic, bins=50, alpha=0.6, density=True, label="Authentic")
    axis.hist(tampered, bins=50, alpha=0.6, density=True, label="Tampered")
    axis.set(
        title="VAE V5 reconstruction-error distributions",
        xlabel="Per-image reconstruction MSE",
        ylabel="Density",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_roc_plot(labels: np.ndarray, mse: np.ndarray, ssim: np.ndarray, path: Path) -> dict[str, float]:
    scores = {"mse": mse, "ssim_error": 1.0 - ssim}
    aucs: dict[str, float] = {}
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, values in scores.items():
        aucs[name] = float(roc_auc_score(labels, values))
        fpr, tpr, _ = roc_curve(labels, values)
        axis.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={aucs[name]:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random ranking")
    axis.set(
        title="Exploratory VAE V5 forensic ranking",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return aucs


def save_generated_samples(model: VAEV5, device: torch.device, path: Path, count: int, seed: int) -> None:
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(count, model.latent_dim, generator=generator, device=device)
        images = model.decode(z).cpu()
    columns = 5
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 2.5 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < count:
            axis.imshow(images[index].permute(1, 2, 0).numpy())
            axis.set_title(f"Synthetic {index + 1}")
        axis.axis("off")
    figure.suptitle("Synthetic VAE V5 research outputs — not genuine forensic evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def calculate_fid(
    model: VAEV5,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    latent_dim: int | None = None,
    sample_count: int | None = None,
    seed: int = 43,
) -> float:
    """Calculate standard Inception-V3 FID with equal real/generated counts."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError("FID requires torchmetrics and torch-fidelity") from exc
    latent_dim = int(latent_dim if latent_dim is not None else model.latent_dim)
    if sample_count is None:
        raise ValueError("sample_count is required")
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            remaining = sample_count - processed
            if remaining <= 0:
                break
            real = batch["image"][:remaining].to(device)
            z = torch.randn(real.size(0), latent_dim, generator=generator, device=device)
            generated = model.decode(z)
            metric.update(real, real=True)
            metric.update(generated, real=False)
            processed += real.size(0)
    if processed != sample_count:
        raise RuntimeError(f"FID processed {processed} images; expected {sample_count}")
    return float(metric.compute().cpu())


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    checkpoint_path = validate_checkpoint_file(args.checkpoint_path, "VAE V5")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_dim = int(checkpoint.get("latent_dim", args.latent_dim))
    beta = float(checkpoint.get("beta", args.beta))
    l1_weight = float(checkpoint.get("l1_weight", args.l1_weight))
    model = VAEV5(latent_dim).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.eval()

    rows: list[dict[str, object]] = []
    grid_samples: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loaders["test"]:
            images = batch["image"].to(device, non_blocking=True)
            reconstruction, mu, logvar = model.reconstruct(images, deterministic=True)
            mse = (reconstruction - images).square().flatten(1).mean(1)
            l1 = (reconstruction - images).abs().flatten(1).mean(1)
            reconstruction_loss = (1.0 - l1_weight) * mse + l1_weight * l1
            kl = -0.5 * torch.sum(1.0 + logvar - mu.square() - logvar.exp(), dim=1)
            total = reconstruction_loss + beta * kl
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            originals_cpu = images.cpu()
            reconstructions_cpu = reconstruction.cpu()

            for index in range(images.size(0)):
                label = int(batch["label"][index])
                ssim = structural_similarity(
                    originals_cpu[index].permute(1, 2, 0).numpy(),
                    reconstructions_cpu[index].permute(1, 2, 0).numpy(),
                    channel_axis=2,
                    data_range=1.0,
                )
                row = {
                    "image_path": str(batch["path"][index]),
                    "label": label,
                    "class_name": str(batch["class_name"][index]),
                    "mse": float(mse[index]),
                    "psnr": float(psnr[index]),
                    "ssim": float(ssim),
                    "ssim_error": float(1.0 - ssim),
                    "reconstruction_loss": float(reconstruction_loss[index]),
                    "kl_loss": float(kl[index]),
                    "total_loss": float(total[index]),
                }
                rows.append(row)
                existing = sum(int(item["label"]) == label for item in grid_samples)
                if existing < args.grid_per_class:
                    grid_samples.append({
                        "label": label,
                        "class_name": row["class_name"],
                        "original": originals_cpu[index],
                        "reconstructed": reconstructions_cpu[index],
                    })

    authentic = [row for row in rows if int(row["label"]) == 0]
    tampered = [row for row in rows if int(row["label"]) == 1]
    actual_counts = (len(rows), len(authentic), len(tampered))
    expected_counts = (args.expected_test_count, args.expected_authentic, args.expected_tampered)
    if actual_counts != expected_counts:
        raise AssertionError(f"Test counts {actual_counts} do not match {expected_counts}")

    args.per_image_path.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grid_samples.sort(key=lambda item: int(item["label"]))
    save_reconstruction_grid(grid_samples, args.grid_path)
    save_error_distribution(rows, args.error_plot_path)
    labels = np.asarray([int(row["label"]) for row in rows])
    mse_values = np.asarray([float(row["mse"]) for row in rows])
    ssim_values = np.asarray([float(row["ssim"]) for row in rows])
    aucs = save_roc_plot(labels, mse_values, ssim_values, args.roc_plot_path)
    save_generated_samples(model, device, args.generated_path, args.generated_samples, args.seed)

    fid: float | None = None
    fid_warning: str | None = "Not calculated; rerun with --run-fid on a suitable GPU."
    if args.run_fid:
        fid = calculate_fid(
            model, loaders["test"], device,
            latent_dim=latent_dim, sample_count=len(rows), seed=args.seed + 1,
        )
        fid_warning = None

    metrics: dict[str, object] = {
        "model": "VAEV5 final mixed-data reconstruction",
        "checkpoint": str(args.checkpoint_path),
        "checkpoint_metadata": {
            key: checkpoint.get(key) for key in (
                "epoch", "val_mse", "val_psnr", "val_l1", "val_kl",
                "latent_dim", "image_size", "l1_weight", "beta",
            )
        },
        "test_images": len(rows),
        "authentic_images": len(authentic),
        "tampered_images": len(tampered),
        "reconstruction_mode": "deterministic_z_equals_mu_with_encoder_skips",
        "overall": summarize(rows),
        "authentic": summarize(authentic),
        "tampered": summarize(tampered),
        "roc_auc": {
            "mse": aucs["mse"],
            "ssim_error": aucs["ssim_error"],
            "interpretation": (
                "Exploratory ranking only; reconstruction error is not a calibrated "
                "tampering probability or standalone detector."
            ),
        },
        "fid": fid,
        "fid_real_samples": len(rows) if fid is not None else 0,
        "fid_generated_samples": len(rows) if fid is not None else 0,
        "fid_warning": fid_warning,
        "evaluation_seconds": time.perf_counter() - started,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/VAE_V5_FINAL.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/vae_v5_final_test_metrics.json"))
    parser.add_argument("--per-image-path", type=Path, default=Path("results/vae_v5_final_test_per_image_metrics.csv"))
    parser.add_argument("--grid-path", type=Path, default=Path("outputs/vae_v5_final/reconstruction_grid.png"))
    parser.add_argument("--error-plot-path", type=Path, default=Path("outputs/vae_v5_final/authentic_vs_tampered_error.png"))
    parser.add_argument("--roc-plot-path", type=Path, default=Path("outputs/vae_v5_final/roc_curves.png"))
    parser.add_argument("--generated-path", type=Path, default=Path("outputs/vae_v5_final/generated_samples.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA_END)
    parser.add_argument("--l1-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-test-count", type=int, default=1892)
    parser.add_argument("--expected-authentic", type=int, default=1123)
    parser.add_argument("--expected-tampered", type=int, default=769)
    parser.add_argument("--grid-per-class", type=int, default=3)
    parser.add_argument("--generated-samples", type=int, default=25)
    parser.add_argument("--run-fid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
