"""Evaluate a trained DCGAN with standard FID and optional Inception Score."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dcgan import DCGANGenerator
from gan_dataset import create_gan_dataloaders


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except ImportError as exc:
        raise RuntimeError("Install torchmetrics[image] and torch-fidelity for GAN evaluation") from exc

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.generator_checkpoint, map_location=device, weights_only=False)
    generator = DCGANGenerator(args.latent_dim).to(device)
    generator.load_state_dict(checkpoint["model_state_dict"]); generator.eval()
    loader = create_gan_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )["test"]
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_metric = InceptionScore(normalize=True, splits=args.inception_splits).to(device)
    random_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    processed = 0; inception_warning: str | None = None

    with torch.no_grad():
        for batch in loader:
            real = ((batch["image"].to(device) + 1.0) / 2.0).clamp(0, 1)
            noise = torch.randn(real.shape[0], args.latent_dim, 1, 1, generator=random_generator, device=device)
            generated = ((generator(noise) + 1.0) / 2.0).clamp(0, 1)
            fid_metric.update(real, real=True); fid_metric.update(generated, real=False)
            try:
                inception_metric.update(generated)
            except Exception as exc:  # IS is optional by requirement.
                inception_warning = f"Inception Score unavailable: {exc}"
            processed += real.shape[0]
    if processed != args.expected_test_count:
        raise AssertionError(f"Evaluated {processed} real/generated images, expected {args.expected_test_count}")
    fid = float(fid_metric.compute().cpu())
    inception_mean = inception_std = None
    if inception_warning is None:
        try:
            score = inception_metric.compute()
            inception_mean, inception_std = float(score[0].cpu()), float(score[1].cpu())
        except Exception as exc:
            inception_warning = f"Inception Score unavailable: {exc}"
    result: dict[str, object] = {
        "real_images": processed, "generated_images": processed,
        "fid": fid, "fid_features": 2048,
        "inception_score_mean": inception_mean, "inception_score_std": inception_std,
        "inception_score_splits": args.inception_splits,
        "evaluation_time_seconds": time.perf_counter() - started,
        "warning": inception_warning,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/best_generator.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/gan_test_metrics.json"))
    parser.add_argument("--image-size", type=int, default=64); parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--expected-test-count", type=int, default=1892)
    parser.add_argument("--inception-splits", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__": print(json.dumps(evaluate(parse_args()), indent=2))
