"""Evaluate Vision Transformer V2 on the canonical 1,892-image test split."""

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
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import AutoencoderImageDataset
from checkpoint_utils import validate_checkpoint_file
from transformer import TransformerForensicAutoencoder

EXPECTED_TEST_COUNTS = {"authentic": 1123, "tampered": 769}


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    """Calculate count plus population mean/std for reconstruction metrics."""
    summary: dict[str, float | int] = {"count": len(rows)}
    for metric in ("mse", "psnr", "ssim"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def relative_image_path(path: str, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


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
        axes[row_index][0].set_title(f"{class_name.title()} — Original Image")
        axes[row_index][1].set_title(f"{class_name.title()} — Reconstructed Image")
        axes[row_index][0].axis("off")
        axes[row_index][1].axis("off")
    figure.suptitle("Vision Transformer V2 — Canonical Held-out Test Reconstructions")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AutoencoderImageDataset(args.splits_dir / "test.csv", args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    checkpoint_path = validate_checkpoint_file(args.checkpoint_path, "Vision Transformer V2")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = (
        "model_state_dict", "image_size", "patch_size", "embed_dim", "num_heads",
        "encoder_layers", "decoder_layers", "ff_dim", "epoch", "validation_mse",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Transformer checkpoint metadata is missing: {missing}")
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError(
            f"Checkpoint image size is {checkpoint['image_size']}, not {args.image_size}."
        )

    model = TransformerForensicAutoencoder(
        image_size=int(checkpoint["image_size"]),
        patch_size=int(checkpoint["patch_size"]),
        embed_dim=int(checkpoint["embed_dim"]),
        num_heads=int(checkpoint["num_heads"]),
        encoder_layers=int(checkpoint["encoder_layers"]),
        decoder_layers=int(checkpoint["decoder_layers"]),
        ff_dim=int(checkpoint["ff_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    rows: list[dict[str, object]] = []
    samples: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "authentic": [],
        "tampered": [],
    }
    inference_seconds = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            inputs = batch["image"].to(device, non_blocking=True)
            inference_started = time.perf_counter()
            reconstructions, _, _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - inference_started
            per_image_mse = torch.mean((reconstructions - inputs) ** 2, dim=(1, 2, 3))

            inputs_cpu = inputs.cpu()
            reconstructions_cpu = reconstructions.cpu()
            for index in range(inputs.shape[0]):
                mse = float(per_image_mse[index])
                psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
                original_hwc = inputs_cpu[index].permute(1, 2, 0).numpy()
                reconstruction_hwc = reconstructions_cpu[index].permute(1, 2, 0).numpy()
                ssim = float(
                    structural_similarity(
                        original_hwc,
                        reconstruction_hwc,
                        data_range=1.0,
                        channel_axis=-1,
                    )
                )
                class_name = str(batch["class_name"][index])
                rows.append(
                    {
                        "image_path": relative_image_path(str(batch["path"][index]), project_root),
                        "label": int(batch["label"][index]),
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
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"Evaluated {len(rows)}/{len(dataset)} images...", flush=True)

    observed_counts = Counter(str(row["class_name"]) for row in rows)
    if len(rows) != sum(EXPECTED_TEST_COUNTS.values()) or dict(observed_counts) != EXPECTED_TEST_COUNTS:
        raise RuntimeError(
            f"Canonical split mismatch: total={len(rows)}, classes={dict(observed_counts)}"
        )
    if not all(
        math.isfinite(float(row[metric]))
        for row in rows
        for metric in ("mse", "psnr", "ssim")
    ):
        raise RuntimeError("Evaluation produced a non-finite metric.")

    args.per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "label", "class_name", "mse", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(rows)

    authentic = [row for row in rows if row["class_name"] == "authentic"]
    tampered = [row for row in rows if row["class_name"] == "tampered"]
    labels = [int(row["label"]) for row in rows]
    mse_scores = [float(row["mse"]) for row in rows]
    metrics: dict[str, object] = {
        "model": "Transformer Forensic Autoencoder V2",
        "checkpoint": args.checkpoint_path.as_posix(),
        "checkpoint_metadata": {
            key: checkpoint.get(key)
            for key in (
                "epoch", "train_mse", "validation_mse", "anomaly_threshold", "roc_auc",
                "balanced_accuracy", "image_size", "patch_size", "num_patches", "embed_dim",
                "num_heads", "encoder_layers", "decoder_layers", "ff_dim",
            )
        },
        "test_images": len(rows),
        "authentic_images": len(authentic),
        "tampered_images": len(tampered),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "standard_deviation_definition": "population standard deviation (ddof=0)",
        "overall": summarize(rows),
        "authentic": summarize(authentic),
        "tampered": summarize(tampered),
        "exploratory_roc_auc": {
            "mse": float(roc_auc_score(labels, mse_scores)),
            "interpretation": (
                "Exploratory ranking only; reconstruction error is not a calibrated tampering "
                "probability or standalone detector."
            ),
        },
        "device": str(device),
        "model_inference_seconds": inference_seconds,
        "evaluation_seconds": time.perf_counter() - started,
        "interpretation_note": (
            "MSE, PSNR, and SSIM measure reconstruction fidelity. Attention and reconstruction "
            "error are not proof or definitive localization of manipulation."
        ),
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_reconstruction_grid(samples, args.reconstruction_grid)
    metrics["evaluation_seconds"] = time.perf_counter() - started
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--checkpoint-path", type=Path, default=Path("checkpoints/transformer_v2_final.pth")
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=Path("results/transformer_v2_test_per_image_metrics.csv"),
    )
    parser.add_argument(
        "--metrics-json", type=Path, default=Path("results/transformer_v2_test_metrics.json")
    )
    parser.add_argument(
        "--reconstruction-grid",
        type=Path,
        default=Path("outputs/transformer_v2/test_reconstruction_grid.png"),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
