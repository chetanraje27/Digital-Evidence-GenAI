"""Evaluate VAE V2 and compare it fairly with the preserved V1 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders
from evaluate_vae import calculate_fid, summarize
from vae import ConvolutionalVAE


def load_model(path: Path, device: torch.device, latent_dim: int) -> ConvolutionalVAE:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConvolutionalVAE(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_comparison_grid(samples: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)), squeeze=False)
    for row_index, sample in enumerate(samples):
        label = str(sample["class_name"]).title()
        for column, (key, title) in enumerate((
            ("original", "Original"), ("v1", "V1 Reconstruction"),
            ("v2", "V2 Reconstruction"),
        )):
            image = sample[key]
            assert isinstance(image, torch.Tensor)
            axes[row_index, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row_index, column].set_title(f"{title} — {label}")
            axes[row_index, column].axis("off")
    figure.suptitle("CASIA Test Images: VAE V1 vs VAE V2 (z = mu)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    loader = loaders["test"]
    v1 = load_model(args.v1_checkpoint, device, args.latent_dim)
    v2 = load_model(args.v2_checkpoint, device, args.latent_dim)
    rows: list[dict[str, object]] = []
    grid: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            v1_mu, _ = v1.encode(inputs)
            v1_reconstructed = v1.decode(v1_mu)
            mu, logvar = v2.encode(inputs)
            reconstructed = v2.decode(mu)
            mse = (reconstructed - inputs).pow(2).flatten(1).mean(1)
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total = mse + args.beta * kl
            inputs_cpu, v1_cpu, v2_cpu = inputs.cpu(), v1_reconstructed.cpu(), reconstructed.cpu()
            for index in range(inputs.shape[0]):
                label = int(batch["label"][index])
                ssim = structural_similarity(
                    inputs_cpu[index].permute(1, 2, 0).numpy(),
                    v2_cpu[index].permute(1, 2, 0).numpy(),
                    channel_axis=2, data_range=1.0,
                )
                rows.append({
                    "image_path": str(batch["path"][index]), "label": label,
                    "class_name": str(batch["class_name"][index]),
                    "mse": float(mse[index]), "psnr": float(psnr[index]),
                    "ssim": float(ssim), "reconstruction_loss": float(mse[index]),
                    "kl_loss": float(kl[index]), "total_loss": float(total[index]),
                })
                if sum(int(item["label"]) == label for item in grid) < args.grid_per_class:
                    grid.append({
                        "label": label, "class_name": str(batch["class_name"][index]),
                        "original": inputs_cpu[index], "v1": v1_cpu[index], "v2": v2_cpu[index],
                    })

    if len(rows) != args.expected_test_count:
        raise AssertionError(f"Evaluated {len(rows)}, expected {args.expected_test_count}")
    save_comparison_grid(sorted(grid, key=lambda row: int(row["label"])), args.grid_path)
    args.per_image_path.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fid = calculate_fid(v2, loader, device, args.latent_dim, len(rows))
    v2_metrics = {
        "test_images": len(rows), "latent_dim": args.latent_dim, "beta": args.beta,
        "reconstruction_mode": "deterministic_z_equals_mu", "overall": summarize(rows),
        "authentic": summarize([row for row in rows if int(row["label"]) == 0]),
        "tampered": summarize([row for row in rows if int(row["label"]) == 1]),
        "fid": fid, "fid_real_samples": len(rows), "fid_generated_samples": len(rows),
        "evaluation_seconds": time.perf_counter() - started,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(v2_metrics, indent=2), encoding="utf-8")

    v1_metrics = json.loads(args.v1_metrics.read_text(encoding="utf-8"))
    comparison = [
        {"version": "V1 baseline", "mse": v1_metrics["overall"]["mse_mean"],
         "psnr": v1_metrics["overall"]["psnr_mean"],
         "ssim": v1_metrics["overall"]["ssim_mean"], "fid": v1_metrics["fid"]},
        {"version": "V2 KL warm-up", "mse": v2_metrics["overall"]["mse_mean"],
         "psnr": v2_metrics["overall"]["psnr_mean"],
         "ssim": v2_metrics["overall"]["ssim_mean"], "fid": fid},
    ]
    with args.comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    return {"v1": comparison[0], "v2": {**comparison[1], "kl": v2_metrics["overall"]["kl_loss_mean"]},
            "evaluation_seconds": v2_metrics["evaluation_seconds"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--v1-checkpoint", type=Path, default=Path("checkpoints/best_vae.pth"))
    parser.add_argument("--v2-checkpoint", type=Path, default=Path("checkpoints/best_vae_v2.pth"))
    parser.add_argument("--v1-metrics", type=Path, default=Path("results/vae_test_metrics.json"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/vae_v2_test_metrics.json"))
    parser.add_argument("--per-image-path", type=Path, default=Path("results/vae_v2_test_per_image_metrics.csv"))
    parser.add_argument("--comparison-path", type=Path, default=Path("results/vae_v1_vs_v2_comparison.csv"))
    parser.add_argument("--grid-path", type=Path, default=Path("outputs/vae/vae_v1_vs_v2_reconstruction.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-test-count", type=int, default=1892)
    parser.add_argument("--grid-per-class", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
