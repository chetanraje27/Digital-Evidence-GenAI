"""Evaluate noisy inputs and DAE outputs against clean held-out test images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoencoder import ConvolutionalAutoencoder
from train_denoising_autoencoder import create_denoising_loaders


def metric_summary(rows: list[dict[str, object]], prefix: str) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[f"{prefix}_{metric}"]) for row in rows]))
        for metric in ("mse", "psnr", "ssim")
    }


def image_metrics(candidate: np.ndarray, clean: np.ndarray) -> tuple[float, float, float]:
    mse = float(np.mean((candidate.astype(np.float64) - clean.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(clean, candidate, data_range=1.0, channel_axis=-1))
    return mse, psnr, ssim


def save_grid(samples: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]], path: Path) -> None:
    figure, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)), squeeze=False)
    for row, (class_name, clean, noisy, denoised) in enumerate(samples):
        for column, (title, image) in enumerate(
            (("Clean", clean), ("Noisy", noisy), ("Denoised", denoised))
        ):
            axes[row][column].imshow(image.permute(1, 2, 0).numpy())
            axes[row][column].set_title(f"{class_name.title()} — {title}")
            axes[row][column].axis("off")
    figure.suptitle("Denoising Autoencoder — Held-out Test Images")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = create_denoising_loaders(args)["test"]
    model = ConvolutionalAutoencoder().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    samples: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sample_counts = Counter()
    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            denoised = model(noisy)
            for index in range(clean.shape[0]):
                clean_hwc = clean[index].cpu().permute(1, 2, 0).numpy()
                noisy_hwc = noisy[index].cpu().permute(1, 2, 0).numpy()
                denoised_hwc = denoised[index].cpu().permute(1, 2, 0).numpy()
                noisy_mse, noisy_psnr, noisy_ssim = image_metrics(noisy_hwc, clean_hwc)
                dae_mse, dae_psnr, dae_ssim = image_metrics(denoised_hwc, clean_hwc)
                class_name = str(batch["class_name"][index])
                rows.append({
                    "image_path": str(batch["path"][index]),
                    "label": int(batch["label"][index]),
                    "class_name": class_name,
                    "noisy_mse": noisy_mse,
                    "noisy_psnr": noisy_psnr,
                    "noisy_ssim": noisy_ssim,
                    "denoised_mse": dae_mse,
                    "denoised_psnr": dae_psnr,
                    "denoised_ssim": dae_ssim,
                })
                if sample_counts[class_name] < args.samples_per_class:
                    samples.append((
                        class_name, clean[index].cpu(), noisy[index].cpu(), denoised[index].cpu()
                    ))
                    sample_counts[class_name] += 1

    assert len(rows) == 1892
    assert Counter(row["class_name"] for row in rows) == Counter({"authentic": 1123, "tampered": 769})
    noisy_summary = metric_summary(rows, "noisy")
    denoised_summary = metric_summary(rows, "denoised")
    improvement = {
        "mse_reduction_percent": (
            (noisy_summary["mse"] - denoised_summary["mse"]) / noisy_summary["mse"] * 100.0
        ),
        "psnr_improvement_db": denoised_summary["psnr"] - noisy_summary["psnr"],
        "ssim_improvement": denoised_summary["ssim"] - noisy_summary["ssim"],
    }

    args.per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics: dict[str, object] = {
        "test_images": len(rows),
        "noise_std": args.noise_std,
        "seed": args.seed,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "noisy_input": noisy_summary,
        "denoised_output": denoised_summary,
        "improvement": improvement,
        "interpretation_note": "Denoising metrics measure noise removal, not forgery detection.",
        "evaluation_time_seconds": time.perf_counter() - start,
    }
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    standard = json.loads(args.standard_metrics.read_text(encoding="utf-8"))["overall"]
    comparison_rows = [
        {"experiment": "Standard Autoencoder", "mse": standard["mse_mean"], "psnr": standard["psnr_mean"], "ssim": standard["ssim_mean"], "compression_ratio": 24, "parameter_count": 265571},
        {"experiment": "Noisy Input", **noisy_summary, "compression_ratio": "N/A", "parameter_count": "N/A"},
        {"experiment": "Denoised Output", **denoised_summary, "compression_ratio": 24, "parameter_count": 265571},
    ]
    with args.comparison_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    save_grid(samples, args.grid_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_denoising_autoencoder.pth"))
    parser.add_argument("--metrics-json", type=Path, default=Path("results/dae_test_metrics.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("results/dae_test_per_image_metrics.csv"))
    parser.add_argument("--comparison-csv", type=Path, default=Path("results/ae_final_comparison.csv"))
    parser.add_argument("--standard-metrics", type=Path, default=Path("results/ae_test_metrics.json"))
    parser.add_argument("--grid-path", type=Path, default=Path("outputs/ae/denoising_comparison_grid.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
