"""Generate a small, reproducible set of presentation-ready Review-2 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ae_inference import AutoencoderInference
from evidence_transformer import DEFAULT_PROMPT, EvidenceTransformer
from transformer_inference import TransformerInference
from vae_inference import VAEInference


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def selected_test_rows() -> list[pd.Series]:
    frame = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    return [
        frame[frame["class_name"] == "authentic"].iloc[0],
        frame[frame["class_name"] == "tampered"].iloc[0],
    ]


def save_ae_examples(output_dir: Path, device: torch.device) -> list[str]:
    model = AutoencoderInference(
        ROOT / "checkpoints" / "best_autoencoder_rtx80_portable.pth", device
    )
    rows = selected_test_rows()
    figure, axes = plt.subplots(2, 2, figsize=(8, 7))
    sources: list[str] = []
    for index, row in enumerate(rows):
        path = ROOT / str(row["image_path"])
        sources.append(path.relative_to(ROOT).as_posix())
        original, reconstructed, metrics = model.reconstruct(load_rgb(path))
        axes[index, 0].imshow(original)
        axes[index, 1].imshow(reconstructed)
        axes[index, 0].set_title(f"{str(row['class_name']).title()} — Original Image")
        axes[index, 1].set_title(
            f"Reconstructed Image\nMSE {metrics['mse']:.5f} · SSIM {metrics['ssim']:.4f}"
        )
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")
    figure.suptitle("Autoencoder RTX-80 — Canonical Test Examples", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "ae_reconstruction_examples.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return sources


def save_vae_variations(output_dir: Path, device: torch.device, source: Path) -> None:
    model = VAEInference(ROOT / "checkpoints" / "VAE_V5_FINAL.pth", device, 256)
    original, deterministic, variations, _ = model.reconstruct_variations(
        load_rgb(source), n_samples=3, temperature=2.0, seed=42
    )
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[1].imshow(deterministic)
    axes[1].set_title("Deterministic\nz = μ")
    for index, variation in enumerate(variations, 2):
        axes[index].imshow(variation["image"])
        axes[index].set_title(
            f"Stochastic {index - 1}\nmean |Δ| {variation['mean_abs_delta_from_deterministic']:.5f}"
        )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        "VAE V5 Image-conditioned Posterior Samples — Temperature 2.0, Seed 42",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "vae_stochastic_variations.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_transformer_example(output_dir: Path, device: torch.device, source: Path) -> None:
    model = TransformerInference(ROOT / "checkpoints" / "transformer_v2_final.pth", device)
    original, reconstructed, metrics, _, attention = model.reconstruct(load_rgb(source))
    overlay = Image.fromarray((attention * 255).astype(np.uint8)).resize(
        (128, 128), Image.Resampling.BILINEAR
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[1].imshow(reconstructed)
    axes[1].set_title(f"Reconstructed Image\nMSE {metrics['mse']:.5f}")
    axes[2].imshow(original.resize((128, 128)))
    axes[2].imshow(overlay, cmap="jet", alpha=0.45)
    axes[2].set_title("Exploratory Attention Overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Vision Transformer V2 — Attention Is Not Tampering Proof", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "vision_transformer_example.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_comparison(output_dir: Path) -> None:
    comparison = json.loads(
        (ROOT / "results" / "image_model_comparison.json").read_text(encoding="utf-8")
    )["models"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = [str(row["model"]).replace(" Final", "") for row in comparison]
    specs = [("mse", "MSE ↓", "#d97706"), ("psnr_db", "PSNR (dB) ↑", "#08758b"), ("ssim", "SSIM ↑", "#16805b")]
    for axis, (key, title, color) in zip(axes, specs):
        values = [float(row[key]) for row in comparison]
        bars = axis.bar(labels, values, color=color, alpha=0.88)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    figure.suptitle("Canonical 1,892-image Reconstruction Comparison", fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "Reconstruction fidelity metrics only — not tampering detection or forensic validity.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(output_dir / "image_model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_text_demo(output_dir: Path, device: torch.device) -> dict[str, object]:
    evidence = (
        "At 09:15 on 12 March 2026, analyst Mira Patel collected device EX-17. "
        "The transfer log records SHA-256 abc123. A later note states that collection began "
        "at 10:15, but no timezone is recorded."
    )
    model = EvidenceTransformer(device=device)
    result = model.generate(evidence, DEFAULT_PROMPT, 128)
    content = (
        "# Pretrained Evidence Transformer — Real Demo Output\n\n"
        "> Synthetic demonstration text; not case evidence.\n\n"
        f"- **Model:** `{result.model_id}`\n"
        f"- **Input tokens:** {result.input_tokens}\n"
        f"- **Output tokens:** {result.output_tokens}\n"
        f"**Inference:** {result.inference_seconds:.3f} seconds on `{device}`\n\n"
        "## Input\n\n"
        f"{evidence}\n\n"
        "## Generated output\n\n"
        f"{result.output}\n"
    )
    (output_dir / "evidence_transformer_demo.md").write_text(content, encoding="utf-8")
    return result.to_dict()


def generate(include_text: bool) -> None:
    output_dir = ROOT / "docs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = save_ae_examples(output_dir, device)
    first_source = ROOT / sources[0]
    test_frame = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    vae_source = ROOT / str(test_frame.iloc[4]["image_path"])
    save_vae_variations(output_dir, device, vae_source)
    save_transformer_example(output_dir, device, first_source)
    save_comparison(output_dir)
    manifest: dict[str, object] = {
        "generated_by": "src/generate_review2_artifacts.py",
        "source_test_images": sources,
        "vae_source_test_image": vae_source.relative_to(ROOT).as_posix(),
        "device": str(device),
        "vae_sampling": {"temperature": 2.0, "seed": 42, "samples": 3},
        "note": (
            "Metrics and images were generated from repository checkpoints and canonical test data. "
            "The fixed VAE example demonstrates posterior sensitivity and is not a population summary."
        ),
    }
    if include_text:
        manifest["text_demo"] = save_text_demo(output_dir, device)
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-text", action="store_true", help="Run cached/downloaded FLAN-T5 demo")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args().include_text)
