"""Evaluate the trained standard Autoencoder on the held-out CASIA test split."""

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

from ae_dataset import create_ae_dataloaders
from autoencoder import ConvolutionalAutoencoder


EXPECTED_TEST_COUNTS = {"authentic": 1123, "tampered": 769}


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    """Return population mean and standard deviation for each image metric."""
    summary: dict[str, float] = {}
    for metric in ("mse", "psnr", "ssim"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def relative_image_path(path: str, project_root: Path) -> str:
    image_path = Path(path).resolve()
    try:
        return image_path.relative_to(project_root).as_posix()
    except ValueError:
        return str(image_path)


def save_reconstruction_grid(
    samples: dict[str, list[tuple[torch.Tensor, torch.Tensor]]], output_path: Path
) -> None:
    pairs = [
        (class_name, original, reconstruction)
        for class_name in ("authentic", "tampered")
        for original, reconstruction in samples[class_name]
    ]
    figure, axes = plt.subplots(len(pairs), 2, figsize=(7, 3 * len(pairs)), squeeze=False)
    for row_index, (class_name, original, reconstruction) in enumerate(pairs):
        axes[row_index][0].imshow(original.permute(1, 2, 0).numpy())
        axes[row_index][1].imshow(reconstruction.permute(1, 2, 0).numpy())
        axes[row_index][0].set_title(f"{class_name.title()} — Original")
        axes[row_index][1].set_title(f"{class_name.title()} — Reconstructed")
        axes[row_index][0].axis("off")
        axes[row_index][1].axis("off")
    figure.suptitle("Standard Autoencoder — Held-out Test Reconstructions")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_distribution_plot(
    rows: list[dict[str, object]], metric: str, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for class_name, color in (("authentic", "tab:blue"), ("tampered", "tab:orange")):
        values = [float(row[metric]) for row in rows if row["class_name"] == class_name]
        axis.hist(values, bins=40, alpha=0.55, density=True, label=class_name.title(), color=color)
        axis.axvline(np.mean(values), color=color, linestyle="--", linewidth=1.5)
    axis.set_xlabel(metric.upper())
    axis.set_ylabel("Density")
    axis.set_title(f"Exploratory Authentic vs Tampered {metric.upper()} Distribution")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    start_time = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = create_ae_dataloaders(
        splits_dir=args.splits_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )["test"]
    model = ConvolutionalAutoencoder().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    samples: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "authentic": [], "tampered": []
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader, start=1):
            inputs = batch["image"].to(device, non_blocking=True)
            reconstructions = model(inputs)
            per_image_mse = torch.mean((reconstructions - inputs) ** 2, dim=(1, 2, 3))

            inputs_cpu = inputs.cpu()
            reconstructions_cpu = reconstructions.cpu()
            for index in range(inputs.shape[0]):
                mse = float(per_image_mse[index].item())
                psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
                original_hwc = inputs_cpu[index].permute(1, 2, 0).numpy()
                reconstruction_hwc = reconstructions_cpu[index].permute(1, 2, 0).numpy()
                ssim = float(
                    structural_similarity(
                        original_hwc, reconstruction_hwc, data_range=1.0, channel_axis=-1
                    )
                )
                class_name = str(batch["class_name"][index])
                rows.append(
                    {
                        "image_path": relative_image_path(str(batch["path"][index]), project_root),
                        "label": int(batch["label"][index].item()),
                        "class_name": class_name,
                        "mse": mse,
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )
                if len(samples[class_name]) < args.samples_per_class:
                    samples[class_name].append(
                        (inputs_cpu[index].clone(), reconstructions_cpu[index].clone())
                    )
            if batch_index % 10 == 0 or batch_index == len(test_loader):
                print(f"Evaluated {len(rows)}/{len(test_loader.dataset)} images...", flush=True)

    observed_counts = Counter(str(row["class_name"]) for row in rows)
    assert len(rows) == 1892
    assert dict(observed_counts) == EXPECTED_TEST_COUNTS
    assert all(math.isfinite(float(row[metric])) for row in rows for metric in ("mse", "psnr", "ssim"))

    args.per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "label", "class_name", "mse", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(rows)

    authentic_rows = [row for row in rows if row["class_name"] == "authentic"]
    tampered_rows = [row for row in rows if row["class_name"] == "tampered"]
    metrics: dict[str, object] = {
        "number_of_test_images": len(rows),
        "number_authentic": len(authentic_rows),
        "number_tampered": len(tampered_rows),
        "compression_ratio": 24,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "standard_deviation_definition": "population standard deviation (ddof=0)",
        "overall": summarize(rows),
        "authentic": summarize(authentic_rows),
        "tampered": summarize(tampered_rows),
        "interpretation_note": (
            "Authentic/tampered results are exploratory reconstruction comparisons only; "
            "the Autoencoder and reconstruction error are not proof of forgery."
        ),
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_reconstruction_grid(samples, args.reconstruction_grid)
    save_distribution_plot(rows, "mse", args.mse_plot)
    save_distribution_plot(rows, "ssim", args.ssim_plot)
    metrics["evaluation_time_seconds"] = time.perf_counter() - start_time
    # Rewrite once to include total evaluation time after generating all figures.
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_autoencoder.pth"))
    parser.add_argument(
        "--per-image-csv", type=Path, default=Path("results/ae_test_per_image_metrics.csv")
    )
    parser.add_argument("--metrics-json", type=Path, default=Path("results/ae_test_metrics.json"))
    parser.add_argument(
        "--reconstruction-grid", type=Path, default=Path("outputs/ae/test_reconstruction_grid.png")
    )
    parser.add_argument(
        "--mse-plot", type=Path, default=Path("outputs/ae/authentic_vs_tampered_mse.png")
    )
    parser.add_argument(
        "--ssim-plot", type=Path, default=Path("outputs/ae/authentic_vs_tampered_ssim.png")
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
