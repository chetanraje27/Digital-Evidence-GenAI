"""Train the standalone DCGAN on the existing CASIA training split."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcgan import build_dcgan
from gan_dataset import create_gan_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def to_display(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0, 1)


def save_samples(generator: nn.Module, fixed_noise: torch.Tensor, path: Path) -> None:
    generator.eval()
    with torch.no_grad(): images = to_display(generator(fixed_noise))
    generator.train()
    figure, axes = plt.subplots(8, 8, figsize=(12, 12))
    for image, axis in zip(images[:64], axes.flat):
        axis.imshow(image.permute(1, 2, 0).numpy()); axis.axis("off")
    figure.suptitle("Synthetic DCGAN-generated samples — not genuine forensic evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=150, bbox_inches="tight"); plt.close(figure)


def save_history(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "generator_loss", "discriminator_loss", "epoch_time_seconds"])
        writer.writeheader(); writer.writerows(rows)


def save_loss_curve(rows: list[dict[str, float | int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    epochs = [row["epoch"] for row in rows]
    axis.plot(epochs, [row["generator_loss"] for row in rows], label="Generator")
    axis.plot(epochs, [row["discriminator_loss"] for row in rows], label="Discriminator")
    axis.set(title="DCGAN Adversarial Training Loss", xlabel="Epoch", ylabel="BCE loss")
    axis.grid(alpha=.3); axis.legend(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = create_gan_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )["train"]
    generator, discriminator = build_dcgan(args.latent_dim)
    generator, discriminator = generator.to(device), discriminator.to(device)
    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    discriminator_optimizer = Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    fixed_generator = torch.Generator(device=device).manual_seed(args.seed)
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, generator=fixed_generator, device=device)
    epoch_limit = 1 if args.smoke_test else args.epochs
    batch_limit = args.smoke_batches if args.smoke_test else None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epoch_limit + 1):
        epoch_started = time.perf_counter(); generator.train(); discriminator.train()
        generator_sum = discriminator_sum = 0.0; image_count = 0
        for batch_index, batch in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit: break
            real = batch["image"].to(device, non_blocking=True)
            size = real.shape[0]
            real_targets = torch.ones(size, 1, 1, 1, device=device)
            fake_targets = torch.zeros(size, 1, 1, 1, device=device)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real), real_targets)
            noise = torch.randn(size, args.latent_dim, 1, 1, device=device)
            fake = generator(noise)
            fake_loss = criterion(discriminator(fake.detach()), fake_targets)
            discriminator_loss = (real_loss + fake_loss) / 2.0
            discriminator_loss.backward(); discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss = criterion(discriminator(fake), real_targets)
            generator_loss.backward(); generator_optimizer.step()

            generator_sum += generator_loss.item() * size
            discriminator_sum += discriminator_loss.item() * size
            image_count += size
        if image_count == 0: raise RuntimeError("No GAN training images were processed")
        row = {
            "epoch": epoch, "generator_loss": generator_sum / image_count,
            "discriminator_loss": discriminator_sum / image_count,
            "epoch_time_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row); save_history(history, args.history_path)
        if epoch % args.sample_interval == 0 or epoch == epoch_limit:
            save_samples(generator, fixed_noise, args.samples_dir / f"generated_epoch_{epoch:02d}.png")
        print(
            f"Epoch {epoch:02d}/{epoch_limit:02d} | G={row['generator_loss']:.6f} | "
            f"D={row['discriminator_loss']:.6f} | seconds={row['epoch_time_seconds']:.1f}", flush=True,
        )

    training_seconds = time.perf_counter() - started
    # GAN loss is adversarial and not a reliable model-selection score. The required
    # best-named files therefore store the completed/final training state explicitly.
    args.generator_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": generator.state_dict(), "epoch": len(history), "config": vars(args)}, args.generator_checkpoint)
    torch.save({"model_state_dict": discriminator.state_dict(), "epoch": len(history), "config": vars(args)}, args.discriminator_checkpoint)
    save_loss_curve(history, args.curve_path)
    save_samples(generator, fixed_noise, args.final_samples_path)
    summary = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs_completed": len(history), "final_generator_loss": history[-1]["generator_loss"],
        "final_discriminator_loss": history[-1]["discriminator_loss"],
        "training_time_seconds": training_seconds,
        "generator_checkpoint": str(args.generator_checkpoint),
        "discriminator_checkpoint": str(args.discriminator_checkpoint),
        "smoke_test": args.smoke_test,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/best_generator.pth"))
    parser.add_argument("--discriminator-checkpoint", type=Path, default=Path("checkpoints/best_discriminator.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/gan_training_history.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/gan_training_summary.json"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/gan/gan_loss_curve.png"))
    parser.add_argument("--samples-dir", type=Path, default=Path("outputs/gan"))
    parser.add_argument("--final-samples-path", type=Path, default=Path("outputs/gan/final_generated_samples.png"))
    parser.add_argument("--image-size", type=int, default=64); parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--learning-rate", type=float, default=.0002)
    parser.add_argument("--beta1", type=float, default=.5); parser.add_argument("--beta2", type=float, default=.999)
    parser.add_argument("--sample-interval", type=int, default=5); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--smoke-batches", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__": print(json.dumps(train(parse_args()), indent=2))
