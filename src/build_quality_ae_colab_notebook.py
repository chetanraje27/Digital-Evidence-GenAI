"""Build the standalone quality-Autoencoder Colab training notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": text.strip().splitlines(True),
    }


cells = [
    markdown("""
# Quality Autoencoder V1 — High-Fidelity Reconstruction

This is a separate deterministic Autoencoder experiment designed for visibly better reconstruction. It uses a residual encoder/decoder, bilinear resize-convolution upsampling, a 6× bottleneck, and a combined L1 + SSIM + edge objective. It has no encoder-to-decoder skip connections.

The existing 24× RTX80 Autoencoder and all earlier checkpoints remain unchanged. The quality model is adopted only after validation selection, full 1,892-image test evaluation, and visual inspection.
"""),
    markdown("## 1. Runtime setup\nSelect **Runtime → Change runtime type → T4 GPU**, then run all cells."),
    code("""
%pip install -q kagglehub Pillow matplotlib numpy pandas scikit-image scikit-learn
"""),
    code("""
import os, sys, json, csv, shutil, subprocess, importlib
from argparse import Namespace
from pathlib import Path, PureWindowsPath
import torch

assert torch.cuda.is_available(), "Enable a GPU runtime: Runtime > Change runtime type > T4 GPU"
GPU_NAME = torch.cuda.get_device_name(0)
print("GPU:", GPU_NAME)
"""),
    markdown("## 2. Pull the current project"),
    code("""
REPO_URL = "https://github.com/chetanraje27/Digital-Evidence-GenAI.git"
PROJECT_ROOT = Path("/content/Digital-Evidence-GenAI")
if not PROJECT_ROOT.exists():
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_ROOT)], check=True)
else:
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"], check=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
print("Commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
"""),
    markdown("## 3. Obtain CASIA v2.0 and validate the canonical manifests"),
    code("""
import kagglehub

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
downloaded = None
if not (RAW_DIR / "CASIA2" / "Au").is_dir():
    downloaded = Path(kagglehub.dataset_download(
        "divg07/casia-20-image-tampering-detection-dataset", output_dir=str(RAW_DIR)
    )).resolve()
if not (RAW_DIR / "CASIA2").is_dir():
    roots = [path for path in (downloaded, RAW_DIR) if path is not None and path.exists()]
    candidates = []
    for root in roots:
        if root.name.casefold() == "casia2" and (root / "Au").is_dir() and (root / "Tp").is_dir():
            candidates.append(root)
        candidates.extend(path for path in root.rglob("CASIA2") if (path / "Au").is_dir() and (path / "Tp").is_dir())
    assert candidates, f"CASIA2/Au and CASIA2/Tp were not found under {roots}"
    os.symlink(candidates[0].resolve(), RAW_DIR / "CASIA2", target_is_directory=True)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
rows_by_split = {}
for split in ("train", "validation", "test"):
    with (SPLITS_DIR / f"{split}.csv").open(newline="", encoding="utf-8") as file:
        rows_by_split[split] = list(csv.DictReader(file))
assert {name: len(rows) for name, rows in rows_by_split.items()} == {"train": 8830, "validation": 1892, "test": 1892}
paths = [row["image_path"] for rows in rows_by_split.values() for row in rows]
assert len(paths) == len(set(path.casefold() for path in paths)) == 12614
assert all(not Path(path).is_absolute() and not PureWindowsPath(path).is_absolute() for path in paths)
missing = [path for path in paths if not (PROJECT_ROOT / path).is_file()]
assert not missing, f"Missing images, first examples: {missing[:5]}"
print("Canonical split validation: PASS")
"""),
    markdown("## 4. Architecture and loss smoke checks"),
    code("""
import quality_autoencoder, train_quality_autoencoder
importlib.reload(quality_autoencoder)
importlib.reload(train_quality_autoencoder)
from quality_autoencoder import QualityAutoencoder
from train_quality_autoencoder import QualityReconstructionLoss

model = QualityAutoencoder()
sample = torch.rand(2, 3, 128, 128)
with torch.inference_mode():
    latent = model.encode(sample)
    reconstructed = model(sample)
assert latent.shape == (2, 32, 16, 16)
assert reconstructed.shape == sample.shape
assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules())
criterion = QualityReconstructionLoss(0.65, 0.25, 0.10)
assert criterion(sample, sample).item() < 1e-5
print("Architecture smoke check: PASS")
print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
print("Latent:", tuple(latent.shape[1:]), "compression: 6x")
"""),
    markdown("""
## 5. Training configuration

Checkpoint selection uses **validation SSIM only**. The held-out test split is not used for training or model selection. Output names are separate from every existing AE checkpoint.
"""),
    code("""
RUN_ROOT = PROJECT_ROOT
TRAIN_ARGS = Namespace(
    splits_dir=SPLITS_DIR,
    checkpoint_path=RUN_ROOT / "checkpoints" / "best_quality_autoencoder_v1.pth",
    history_path=RUN_ROOT / "results" / "quality_ae_v1_training_history.csv",
    summary_path=RUN_ROOT / "results" / "quality_ae_v1_training_summary.json",
    curve_path=RUN_ROOT / "outputs" / "quality_ae_v1" / "training_curve.png",
    grid_path=RUN_ROOT / "outputs" / "quality_ae_v1" / "validation_reconstruction_grid.png",
    image_size=128, batch_size=32, num_workers=2,
    learning_rate=3e-4, min_learning_rate=1e-6, weight_decay=1e-5,
    max_epochs=80, patience=12, lr_patience=4, lr_factor=0.5, min_delta=1e-4,
    l1_weight=0.65, ssim_weight=0.25, edge_weight=0.10,
    seed=42, smoke_test=False, smoke_batches=2, require_cuda=True,
)
print(vars(TRAIN_ARGS))
"""),
    markdown("## 6. Mandatory two-batch smoke run"),
    code("""
smoke_dir = Path("/content/quality_ae_smoke")
SMOKE_ARGS = Namespace(**{**vars(TRAIN_ARGS),
    "checkpoint_path": smoke_dir / "model.pth",
    "history_path": smoke_dir / "history.csv",
    "summary_path": smoke_dir / "summary.json",
    "curve_path": smoke_dir / "curve.png",
    "grid_path": smoke_dir / "grid.png",
    "smoke_test": True,
})
smoke_summary = train_quality_autoencoder.train(SMOKE_ARGS)
assert smoke_summary["epochs_completed"] == 1
saved_smoke = torch.load(SMOKE_ARGS.checkpoint_path, map_location="cpu", weights_only=True)
assert saved_smoke["architecture"] == "quality_residual_ae_v1"
print("Training/checkpoint smoke test: PASS")
"""),
    markdown("## 7. Full GPU training"),
    code("""
RUN_FULL_TRAINING = True
if RUN_FULL_TRAINING:
    training_summary = train_quality_autoencoder.train(TRAIN_ARGS)
    print(json.dumps(training_summary, indent=2))
else:
    print("Training skipped.")
"""),
    markdown("## 8. Inspect validation history and reconstructions"),
    code("""
import pandas as pd
from IPython.display import display, Image as DisplayImage

history = pd.read_csv(TRAIN_ARGS.history_path)
display(history.tail())
display(DisplayImage(filename=str(TRAIN_ARGS.curve_path)))
display(DisplayImage(filename=str(TRAIN_ARGS.grid_path)))
best = torch.load(TRAIN_ARGS.checkpoint_path, map_location="cpu", weights_only=True)
print("Selected epoch:", best["epoch"])
print("Validation SSIM:", best["validation_ssim"])
print("Validation MSE:", best["validation_mse"])
"""),
    markdown("## 9. One-time canonical held-out test evaluation"),
    code("""
import evaluate_autoencoder
importlib.reload(evaluate_autoencoder)

EVAL_ARGS = Namespace(
    splits_dir=SPLITS_DIR,
    checkpoint_path=TRAIN_ARGS.checkpoint_path,
    per_image_csv=PROJECT_ROOT / "results" / "quality_ae_v1_test_per_image_metrics.csv",
    metrics_json=PROJECT_ROOT / "results" / "quality_ae_v1_test_metrics.json",
    reconstruction_grid=PROJECT_ROOT / "outputs" / "quality_ae_v1" / "test_reconstruction_grid.png",
    mse_plot=PROJECT_ROOT / "outputs" / "quality_ae_v1" / "authentic_vs_tampered_mse.png",
    ssim_plot=PROJECT_ROOT / "outputs" / "quality_ae_v1" / "authentic_vs_tampered_ssim.png",
    image_size=128, batch_size=32, num_workers=2, samples_per_class=3, seed=42,
)
test_metrics = evaluate_autoencoder.evaluate(EVAL_ARGS)
print(json.dumps(test_metrics, indent=2))
display(DisplayImage(filename=str(EVAL_ARGS.reconstruction_grid)))
"""),
    markdown("## 10. Honest comparison with both existing AE results"),
    code("""
comparison_rows = []
for label, filename in (
    ("RTX80 baseline (24x)", "ae_rtx80_test_metrics.json"),
    ("RTX80 fine-tuned (24x)", "ae_rtx80_finetuned_test_metrics.json"),
):
    path = PROJECT_ROOT / "results" / filename
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))
        comparison_rows.append({"model": label, **values["overall"]})
comparison_rows.append({"model": "Quality AE V1 (6x)", **test_metrics["overall"]})
comparison = pd.DataFrame(comparison_rows)
display(comparison[["model", "mse_mean", "psnr_mean", "ssim_mean"]])
baseline = next((row for row in comparison_rows if row["model"].startswith("RTX80 baseline")), None)
if baseline:
    print("Quality AE improves MSE:", test_metrics["overall"]["mse_mean"] < baseline["mse_mean"])
    print("Quality AE improves PSNR:", test_metrics["overall"]["psnr_mean"] > baseline["psnr_mean"])
    print("Quality AE improves SSIM:", test_metrics["overall"]["ssim_mean"] > baseline["ssim_mean"])
print("Visual inspection is still required before adoption.")
print("Reconstruction error is not a tampering probability or forensic verdict.")
"""),
    markdown("## 11. Save every required artifact to Google Drive"),
    code("""
from google.colab import drive
drive.mount("/content/drive")
DRIVE_PROJECT = Path("/content/drive/MyDrive/Digital_Evidence")
artifact_paths = [
    TRAIN_ARGS.checkpoint_path, TRAIN_ARGS.history_path, TRAIN_ARGS.summary_path,
    TRAIN_ARGS.curve_path, TRAIN_ARGS.grid_path,
    EVAL_ARGS.per_image_csv, EVAL_ARGS.metrics_json, EVAL_ARGS.reconstruction_grid,
    EVAL_ARGS.mse_plot, EVAL_ARGS.ssim_plot,
]
for source in artifact_paths:
    if not source.is_file():
        raise FileNotFoundError(f"Required artifact was not produced: {source}")
    destination = DRIVE_PROJECT / source.relative_to(PROJECT_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print("Saved:", destination)
"""),
    markdown("## 12. Final measured report"),
    code("""
print("GPU:", GPU_NAME)
print("Architecture:", training_summary["architecture"])
print("Compression ratio:", training_summary["compression_ratio"])
print("Best epoch:", training_summary["best_epoch"])
print("Validation MSE:", training_summary["best_validation_mse"])
print("Validation SSIM:", training_summary["best_validation_ssim"])
print("Test MSE:", test_metrics["overall"]["mse_mean"])
print("Test PSNR:", test_metrics["overall"]["psnr_mean"])
print("Test SSIM:", test_metrics["overall"]["ssim_mean"])
print("Checkpoint:", TRAIN_ARGS.checkpoint_path)
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "04_quality_autoencoder_training_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "notebooks" / "AE" / "04_quality_autoencoder_training_colab.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(target)
