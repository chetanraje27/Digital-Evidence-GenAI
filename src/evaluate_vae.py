"""Final evaluation and visualizations for the trained CASIA convolutional VAE."""

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
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from vae import ConvolutionalVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"count": len(rows)}
    for key in ("mse", "psnr", "ssim", "reconstruction_loss", "kl_loss", "total_loss"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=0))
    return result


def save_reconstruction_grid(samples: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(len(samples), 2, figsize=(7, 3 * len(samples)), squeeze=False)
    for row_index, sample in enumerate(samples):
        original = sample["original"]
        reconstructed = sample["reconstructed"]
        assert isinstance(original, torch.Tensor) and isinstance(reconstructed, torch.Tensor)
        axes[row_index, 0].imshow(original.permute(1, 2, 0).numpy())
        axes[row_index, 1].imshow(reconstructed.permute(1, 2, 0).numpy())
        label = str(sample["class_name"]).title()
        axes[row_index, 0].set_title(f"Original — {label}")
        axes[row_index, 1].set_title(f"Reconstructed — {label}")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle("VAE Test Reconstructions (deterministic z = mu)", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_generated_samples(
    model: ConvolutionalVAE, device: torch.device, latent_dim: int, path: Path,
    count: int = 25,
) -> None:
    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        z = torch.randn(count, latent_dim, generator=generator, device=device)
        images = model.decode(z).cpu()
    columns = 5
    figure, axes = plt.subplots(math.ceil(count / columns), columns, figsize=(12, 12), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < count:
            axis.imshow(images[index].permute(1, 2, 0).numpy())
            axis.set_title(f"Synthetic {index + 1}")
        axis.axis("off")
    figure.suptitle("Synthetic VAE-generated samples", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_interpolation(
    model: ConvolutionalVAE, endpoints: list[torch.Tensor], device: torch.device,
    path: Path, steps: int = 10,
) -> None:
    inputs = torch.stack(endpoints).to(device)
    with torch.no_grad():
        mu, _ = model.encode(inputs)
        weights = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
        latent = (1 - weights) * mu[0] + weights * mu[1]
        decoded = model.decode(latent).cpu()
    figure, axes = plt.subplots(1, steps, figsize=(20, 3))
    for index, axis in enumerate(axes):
        axis.imshow(decoded[index].permute(1, 2, 0).numpy())
        axis.set_title(f"t={index/(steps-1):.2f}")
        axis.axis("off")
    figure.suptitle("VAE latent interpolation (authentic endpoint → tampered endpoint)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def calculate_fid(
    model: ConvolutionalVAE, loader: torch.utils.data.DataLoader,
    device: torch.device, latent_dim: int, sample_count: int,
) -> float:
    """Compute standard Inception-V3 FID with equal real and generated counts."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError("FID requires: pip install torchmetrics[image] torch-fidelity") from exc
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    generator = torch.Generator(device=device).manual_seed(4242)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            real = batch["image"].to(device)
            remaining = sample_count - processed
            if remaining <= 0:
                break
            real = real[:remaining]
            generated = model.decode(torch.randn(
                real.shape[0], latent_dim, generator=generator, device=device
            ))
            metric.update(real, real=True)
            metric.update(generated, real=False)
            processed += real.shape[0]
    if processed != sample_count:
        raise RuntimeError(f"FID processed {processed}, expected {sample_count}")
    return float(metric.compute().cpu())


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    loader = loaders["test"]
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model = ConvolutionalVAE(args.latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    grid_samples: list[dict[str, object]] = []
    endpoints: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            mu, logvar = model.encode(inputs)
            reconstructions = model.decode(mu)  # deterministic evaluation
            mse = (reconstructions - inputs).pow(2).flatten(1).mean(1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total = mse + args.beta * kl
            originals_cpu = inputs.cpu()
            reconstructed_cpu = reconstructions.cpu()
            for index in range(inputs.shape[0]):
                label = int(batch["label"][index])
                class_name = str(batch["class_name"][index])
                ssim = structural_similarity(
                    originals_cpu[index].permute(1, 2, 0).numpy(),
                    reconstructed_cpu[index].permute(1, 2, 0).numpy(),
                    channel_axis=2, data_range=1.0,
                )
                rows.append({
                    "image_path": str(batch["path"][index]), "label": label,
                    "class_name": class_name, "mse": float(mse[index]),
                    "psnr": float(psnr[index]), "ssim": float(ssim),
                    "reconstruction_loss": float(mse[index]), "kl_loss": float(kl[index]),
                    "total_loss": float(total[index]),
                })
                if label not in endpoints:
                    endpoints[label] = originals_cpu[index]
                class_grid_count = sum(int(item["label"]) == label for item in grid_samples)
                if class_grid_count < args.grid_per_class:
                    grid_samples.append({
                        "label": label, "class_name": class_name,
                        "original": originals_cpu[index],
                        "reconstructed": reconstructed_cpu[index],
                    })

    if len(rows) != args.expected_test_count:
        raise AssertionError(f"Evaluated {len(rows)} images, expected {args.expected_test_count}")
    if set(endpoints) != {0, 1}:
        raise AssertionError("Both authentic and tampered interpolation endpoints are required")

    args.per_image_path.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_reconstruction_grid(sorted(grid_samples, key=lambda item: int(item["label"])), args.grid_path)
    save_generated_samples(model, device, args.latent_dim, args.generated_path, args.display_samples)
    save_interpolation(model, [endpoints[0], endpoints[1]], device, args.interpolation_path, args.interpolation_steps)

    fid = calculate_fid(model, loader, device, args.latent_dim, len(rows))
    authentic = [row for row in rows if int(row["label"]) == 0]
    tampered = [row for row in rows if int(row["label"]) == 1]
    metrics: dict[str, object] = {
        "test_images": len(rows), "authentic_images": len(authentic),
        "tampered_images": len(tampered), "latent_dim": args.latent_dim,
        "beta": args.beta, "reconstruction_mode": "deterministic_z_equals_mu",
        "overall": summarize(rows), "authentic": summarize(authentic),
        "tampered": summarize(tampered), "fid": fid,
        "fid_real_samples": len(rows), "fid_generated_samples": len(rows),
        "fid_implementation": "torchmetrics.image.fid.FrechetInceptionDistance(feature=2048)",
        "evaluation_seconds": time.perf_counter() - started,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_vae.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/vae_test_metrics.json"))
    parser.add_argument("--per-image-path", type=Path, default=Path("results/vae_test_per_image_metrics.csv"))
    parser.add_argument("--grid-path", type=Path, default=Path("outputs/vae/vae_test_reconstruction_grid.png"))
    parser.add_argument("--generated-path", type=Path, default=Path("outputs/vae/vae_generated_samples.png"))
    parser.add_argument("--interpolation-path", type=Path, default=Path("outputs/vae/vae_latent_interpolation.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-test-count", type=int, default=1892)
    parser.add_argument("--grid-per-class", type=int, default=3)
    parser.add_argument("--display-samples", type=int, default=25)
    parser.add_argument("--interpolation-steps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2))
