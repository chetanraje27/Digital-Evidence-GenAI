"""Build the concise, reproducible AE V2 Colab training notebook."""

from pathlib import Path
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(True)}


cells = [
md("""
# Autoencoder V2 — Quality-Focused CASIA Training on Colab

This notebook retrains/fine-tunes the **existing validated convolutional Autoencoder**; it does not change the architecture, 24× bottleneck, dataset split, or `[0,1]` preprocessing.

Recommended strategy: initialize from `best_autoencoder.pth` (epoch 42 baseline), then fine-tune with a lower learning rate. The new run writes only `*_v2` artifacts, so the baseline remains recoverable. Better results are selected strictly by validation MSE—improvement is targeted, not guaranteed.
"""),
md("## 1. Colab setup\nSelect **Runtime → Change runtime type → T4 GPU** before running."),
code("""
%pip install -q kagglehub Pillow matplotlib numpy pandas scikit-image scikit-learn
"""),
code("""
import os, sys, json, csv, shutil, subprocess, importlib
from pathlib import Path, PureWindowsPath
import torch

assert torch.cuda.is_available(), "Enable a GPU runtime: Runtime > Change runtime type > T4 GPU"
GPU_NAME = torch.cuda.get_device_name(0)
print("CUDA:", torch.cuda.is_available())
print("GPU:", GPU_NAME)
"""),
md("## 2. Obtain project code\nThe CASIA images and trained weights are not uploaded to GitHub."),
code("""
REPO_URL = "https://github.com/chetanraje27/Digital-Evidence-GenAI.git"
PROJECT_ROOT = Path("/content/Digital-Evidence-GenAI")
if not PROJECT_ROOT.exists():
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_ROOT)], check=True)
else:
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"], check=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
print("Project:", PROJECT_ROOT)
"""),
md("## 3. Download CASIA v2.0 and validate paths\nThe committed manifests remain the source of truth. Ground-truth PNG masks are excluded."),
code("""
import kagglehub

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
if not (RAW_DIR / "CASIA2" / "Au").is_dir():
    downloaded = Path(kagglehub.dataset_download(
        "divg07/casia-20-image-tampering-detection-dataset", output_dir=str(RAW_DIR)
    ))
    print("Downloaded to:", downloaded)

# Locate CASIA2 if KaggleHub added an extra directory level, then expose the
# repository-relative location required by the portable CSV manifests.
if not (RAW_DIR / "CASIA2").is_dir():
    candidates = [p for p in RAW_DIR.rglob("CASIA2") if (p / "Au").is_dir() and (p / "Tp").is_dir()]
    assert candidates, "CASIA2/Au and CASIA2/Tp were not found after download."
    os.symlink(candidates[0], RAW_DIR / "CASIA2", target_is_directory=True)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
rows_by_split = {}
for name in ("train", "validation", "test"):
    with (SPLITS_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        rows_by_split[name] = list(csv.DictReader(f))
all_rows = sum(rows_by_split.values(), [])
paths = [row["image_path"] for row in all_rows]
assert {name: len(rows) for name, rows in rows_by_split.items()} == {"train": 8830, "validation": 1892, "test": 1892}
assert len(paths) == len(set(p.casefold() for p in paths)) == 12614
assert all(not Path(p).is_absolute() and not PureWindowsPath(p).is_absolute() for p in paths)
assert all(Path(p).suffix.lower() != ".png" and "groundtruth" not in p.lower() for p in paths)
missing = [p for p in paths if not (PROJECT_ROOT / p).is_file()]
assert not missing, f"Missing images, first examples: {missing[:5]}"
print("Split and leakage validation: PASS", {k: len(v) for k, v in rows_by_split.items()})
"""),
md("## 4. Validate pipeline and architecture"),
code("""
from ae_dataset import create_ae_dataloaders
from autoencoder import ConvolutionalAutoencoder

loaders = create_ae_dataloaders(SPLITS_DIR, image_size=128, batch_size=32, num_workers=2, seed=42)
batch = next(iter(loaders["train"]))["image"]
model = ConvolutionalAutoencoder()
with torch.inference_mode():
    latent = model.encode(batch[:2])
    output = model(batch[:2])
assert batch.shape == (32, 3, 128, 128)
assert latent.shape == (2, 32, 8, 8) and output.shape == (2, 3, 128, 128)
assert 0 <= batch.min() <= batch.max() <= 1
print("Batch:", tuple(batch.shape), "range:", (float(batch.min()), float(batch.max())))
print("Latent:", tuple(latent.shape), "parameters:", sum(p.numel() for p in model.parameters()), "compression: 24x")
"""),
md("""
## 5. Baseline checkpoint

Recommended: place the existing `best_autoencoder.pth` in Google Drive at `/content/drive/MyDrive/Digital_Evidence/checkpoints/`. If it is absent, the notebook can train from scratch, but fine-tuning the validated baseline is the safer route to improving its result.
"""),
code("""
from google.colab import drive
drive.mount("/content/drive")

DRIVE_PROJECT = Path("/content/drive/MyDrive/Digital_Evidence")
DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
baseline_candidates = [
    DRIVE_PROJECT / "checkpoints" / "best_autoencoder.pth",
    PROJECT_ROOT / "checkpoints" / "best_autoencoder.pth",
]
BASELINE_CHECKPOINT = next((p for p in baseline_candidates if p.is_file()), None)
print("Baseline checkpoint:", BASELINE_CHECKPOINT or "NOT FOUND — scratch training will be used")
if BASELINE_CHECKPOINT:
    ckpt = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=True)
    print("Baseline epoch:", ckpt.get("epoch"), "validation MSE:", ckpt.get("validation_loss"))
"""),
md("""
## 6. Quality-focused configuration

- Existing architecture, 128×128 input, batch size 32, Adam, and MSE remain unchanged.
- Fine-tuning starts at `1e-4`; `ReduceLROnPlateau` can reduce it to `1e-6`.
- Up to 60 additional epochs; early stopping patience 10.
- CUDA mixed precision improves T4 speed without changing the model.
"""),
code("""
from argparse import Namespace

V2_ROOT = PROJECT_ROOT
common = dict(
    splits_dir=SPLITS_DIR, image_size=128, batch_size=32, num_workers=2,
    learning_rate=1e-4 if BASELINE_CHECKPOINT else 1e-3,
    min_learning_rate=1e-6, weight_decay=1e-6, max_epochs=60,
    patience=10, lr_patience=3, lr_factor=0.5, min_delta=1e-7,
    seed=42, initial_checkpoint=BASELINE_CHECKPOINT,
)
TRAIN_ARGS = Namespace(**common,
    checkpoint_path=V2_ROOT / "checkpoints" / "best_autoencoder_v2.pth",
    history_path=V2_ROOT / "results" / "ae_v2_training_history.csv",
    summary_path=V2_ROOT / "results" / "ae_v2_training_summary.json",
    curve_path=V2_ROOT / "outputs" / "ae_v2" / "training_curve.png",
    grid_path=V2_ROOT / "outputs" / "ae_v2" / "reconstruction_grid.png",
    smoke_test=False, smoke_batches=2, require_cuda=True,
)
print(vars(TRAIN_ARGS))
"""),
md("## 7. Required smoke test\nThis uses two train/validation batches and writes temporary artifacts only."),
code("""
import train_autoencoder_v2
importlib.reload(train_autoencoder_v2)

smoke_dir = Path("/content/ae_v2_smoke")
SMOKE_ARGS = Namespace(**{**vars(TRAIN_ARGS),
    "checkpoint_path": smoke_dir / "smoke.pth", "history_path": smoke_dir / "history.csv",
    "summary_path": smoke_dir / "summary.json", "curve_path": smoke_dir / "curve.png",
    "grid_path": smoke_dir / "grid.png", "smoke_test": True, "require_cuda": True,
})
smoke_summary = train_autoencoder_v2.train(SMOKE_ARGS)
assert smoke_summary["epochs_completed"] == 1
print("Smoke test: PASS", smoke_summary)
"""),
md("## 8. Full GPU training\nSet `RUN_FULL_TRAINING = True`. The best validation checkpoint is saved, not merely the final epoch."),
code("""
RUN_FULL_TRAINING = True
if RUN_FULL_TRAINING:
    training_summary = train_autoencoder_v2.train(TRAIN_ARGS)
    print(json.dumps(training_summary, indent=2))
else:
    print("Full training skipped.")
"""),
md("## 9. Training curves and best checkpoint"),
code("""
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage

history = pd.read_csv(TRAIN_ARGS.history_path)
display(history.tail())
display(DisplayImage(filename=str(TRAIN_ARGS.curve_path)))
display(DisplayImage(filename=str(TRAIN_ARGS.grid_path)))
best = torch.load(TRAIN_ARGS.checkpoint_path, map_location="cpu", weights_only=True)
print("Best fine-tuning epoch:", best.get("epoch"))
print("Best validation MSE:", best.get("validation_loss"))
"""),
md("## 10. Complete held-out test evaluation\nEvaluation uses all 1,892 test images and reports per-image MSE, PSNR, and SSIM."),
code("""
from evaluate_autoencoder import evaluate

EVAL_ARGS = Namespace(
    splits_dir=SPLITS_DIR, checkpoint_path=TRAIN_ARGS.checkpoint_path,
    per_image_csv=PROJECT_ROOT / "results" / "ae_v2_test_per_image_metrics.csv",
    metrics_json=PROJECT_ROOT / "results" / "ae_v2_test_metrics.json",
    reconstruction_grid=PROJECT_ROOT / "outputs" / "ae_v2" / "test_reconstruction_grid.png",
    mse_plot=PROJECT_ROOT / "outputs" / "ae_v2" / "authentic_vs_tampered_mse.png",
    ssim_plot=PROJECT_ROOT / "outputs" / "ae_v2" / "authentic_vs_tampered_ssim.png",
    image_size=128, batch_size=32, num_workers=2, samples_per_class=3, seed=42,
)
test_metrics = evaluate(EVAL_ARGS)
print(json.dumps(test_metrics, indent=2))
"""),
md("## 11. Visual test results and baseline comparison"),
code("""
display(DisplayImage(filename=str(EVAL_ARGS.reconstruction_grid)))
display(DisplayImage(filename=str(EVAL_ARGS.mse_plot)))

baseline_metrics_path = PROJECT_ROOT / "results" / "ae_test_metrics.json"
if baseline_metrics_path.is_file():
    baseline_metrics = json.loads(baseline_metrics_path.read_text())
    comparison = pd.DataFrame([
        {"model": "Current baseline", **{m: baseline_metrics["overall"][f"{m}_mean"] for m in ("mse", "psnr", "ssim")}},
        {"model": "New AE V2", **{m: test_metrics["overall"][f"{m}_mean"] for m in ("mse", "psnr", "ssim")}},
    ])
    display(comparison)
    print("V2 improved MSE:", comparison.loc[1, "mse"] < comparison.loc[0, "mse"])
else:
    print("Baseline metrics file unavailable; V2 metrics are shown above.")
print("Note: authentic/tampered reconstruction differences are exploratory; the AE is not a forgery detector.")
"""),
md("## 12. Persist V2 artifacts to Google Drive"),
code("""
artifact_paths = [
    TRAIN_ARGS.checkpoint_path, TRAIN_ARGS.history_path, TRAIN_ARGS.summary_path,
    TRAIN_ARGS.curve_path, TRAIN_ARGS.grid_path, EVAL_ARGS.per_image_csv,
    EVAL_ARGS.metrics_json, EVAL_ARGS.reconstruction_grid, EVAL_ARGS.mse_plot, EVAL_ARGS.ssim_plot,
]
for source in artifact_paths:
    relative = source.relative_to(PROJECT_ROOT)
    destination = DRIVE_PROJECT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print("Saved:", destination)
"""),
md("## 13. Final report"),
code("""
print("GPU:", GPU_NAME)
print("Epochs completed:", training_summary["epochs_completed"])
print("Best fine-tuning epoch:", training_summary["best_epoch"])
print("Best validation MSE:", training_summary["best_validation_loss"])
print("Test MSE:", test_metrics["overall"]["mse_mean"])
print("Test PSNR:", test_metrics["overall"]["psnr_mean"])
print("Test SSIM:", test_metrics["overall"]["ssim_mean"])
print("Early stopping:", training_summary["early_stopping_triggered"])
print("Training seconds:", training_summary["training_time_seconds"])
print("Checkpoint:", TRAIN_ARGS.checkpoint_path)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"name": "02_autoencoder_training_colab.ipynb"},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
target = Path(__file__).resolve().parents[1] / "notebooks" / "AE" / "02_autoencoder_training_colab.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(target)
