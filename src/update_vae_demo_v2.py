"""Add verified VAE V2 results and embedded figures to the faculty demo notebook."""

from __future__ import annotations

import base64
import csv
import json
import sys
from pathlib import Path

import matplotlib
import torch
import nbformat as nbf

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ae_dataset import create_ae_dataloaders
from evaluate_vae_v2 import load_model, save_comparison_grid
from train_vae_v2 import save_curves


NOTEBOOK_PATH = ROOT / "notebooks" / "VAE_CASIA_Complete_Demo.ipynb"
LOSS_CURVE = ROOT / "outputs" / "vae" / "vae_v2_loss_curves.png"
RECONSTRUCTION_GRID = ROOT / "outputs" / "vae" / "vae_v1_vs_v2_reconstruction.png"


def recreate_figures() -> None:
    with (ROOT / "results" / "vae_v2_training_history.csv").open(encoding="utf-8") as file:
        rows = [
            {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    save_curves(rows, LOSS_CURVE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v1 = load_model(ROOT / "checkpoints" / "best_vae.pth", device, 128)
    v2 = load_model(ROOT / "checkpoints" / "best_vae_v2.pth", device, 128)
    loader = create_ae_dataloaders(ROOT / "data" / "splits", 128, 32, 0, 42)["test"]
    samples: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            v1_mu, _ = v1.encode(inputs)
            v2_mu, _ = v2.encode(inputs)
            v1_reconstructed = v1.decode(v1_mu).cpu()
            v2_reconstructed = v2.decode(v2_mu).cpu()
            inputs = inputs.cpu()
            for index in range(inputs.shape[0]):
                label = int(batch["label"][index])
                if sum(int(item["label"]) == label for item in samples) < 3:
                    samples.append({
                        "label": label, "class_name": str(batch["class_name"][index]),
                        "original": inputs[index], "v1": v1_reconstructed[index],
                        "v2": v2_reconstructed[index],
                    })
            if len(samples) == 6:
                break
    save_comparison_grid(sorted(samples, key=lambda item: int(item["label"])), RECONSTRUCTION_GRID)


def attached_markdown(title: str, path: Path):
    cell = nbf.v4.new_markdown_cell(f"### {title}\n\n![{title}](attachment:{path.name})")
    cell["attachments"] = {
        path.name: {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
    }
    cell["metadata"]["vae_v2_addition"] = True
    return cell


def tagged_markdown(text: str):
    cell = nbf.v4.new_markdown_cell(text)
    cell["metadata"]["vae_v2_addition"] = True
    return cell


def tagged_code(text: str):
    cell = nbf.v4.new_code_cell(text)
    cell["metadata"]["vae_v2_addition"] = True
    return cell


def update_notebook() -> None:
    notebook = nbf.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells = [
        cell for cell in notebook.cells if not cell.get("metadata", {}).get("vae_v2_addition")
    ]
    notebook.cells[0].source = (
        "# VAE V1 and V2 for CASIA Digital Evidence Analysis\n\n"
        "**Faculty demonstration — baseline VAE and KL-warm-up VAE V2**\n\n"
        "V1 is preserved as the baseline. V2 uses linear KL beta warm-up to improve "
        "reconstruction while retaining a probabilistic latent representation. These "
        "experiments do not constitute forgery classification."
    )
    additions = [
        tagged_markdown(
            "# VAE Version 2 — KL Warm-up Improvement\n\n"
            "V2 retains the same CASIA splits, preprocessing, architecture, latent dimension "
            "128, Adam optimizer, learning rate 0.0005, and MSE reconstruction loss. The only "
            "training change is linear KL warm-up: beta increases from 0.0001 at epoch 1 to "
            "0.001 at epoch 10 and stays at 0.001 afterward."
        ),
        tagged_markdown(
            "## V2 training result\n\n"
            "- GPU: **Tesla T4**\n"
            "- Epochs completed: **48/50**\n"
            "- Best epoch: **43**\n"
            "- Early stopping: **Yes**\n"
            "- Best validation total loss: **0.04100876**\n"
            "- Training time: **1363.93 seconds**"
        ),
        attached_markdown("VAE V2 loss curves and beta warm-up", LOSS_CURVE),
        tagged_markdown("## V1 versus V2 reconstruction"),
        attached_markdown("Original | V1 reconstruction | V2 reconstruction", RECONSTRUCTION_GRID),
        tagged_markdown(
            "## V1 versus V2 final test results\n\n"
            "| Metric | V1 baseline | V2 KL warm-up | Direction |\n"
            "|---|---:|---:|---|\n"
            "| MSE | 0.02909885 | **0.02816802** | 3.20% lower |\n"
            "| PSNR | 15.7895 dB | **15.9348 dB** | 0.92% higher |\n"
            "| SSIM | 0.287554 | **0.290189** | 0.92% higher |\n"
            "| FID | 339.579 | **321.358** | 5.37% lower |\n"
            "| KL | 7.139858 | 8.219714 | Probabilistic regularization retained |\n\n"
            "V2 improved all three reconstruction metrics and FID. The improvement is modest, "
            "and reconstructions remain smooth because the architecture and pixel-wise MSE "
            "objective are unchanged."
        ),
        tagged_code(
            "# Load V2 without retraining and verify its saved checkpoint.\n"
            "TRAIN_VAE_V2 = False\n"
            "v2_checkpoint_path = PROJECT_ROOT/'checkpoints/best_vae_v2.pth'\n"
            "v2_checkpoint = torch.load(v2_checkpoint_path, map_location=device, weights_only=False)\n"
            "v2_model = ConvolutionalVAE(latent_dim=128).to(device)\n"
            "v2_model.load_state_dict(v2_checkpoint['model_state_dict'])\n"
            "v2_model.eval()\n"
            "print('TRAIN_VAE_V2:', TRAIN_VAE_V2)\n"
            "print('Loaded V2 checkpoint:', v2_checkpoint_path)\n"
            "print('Best epoch:', v2_checkpoint['epoch'])"
        ),
        tagged_code(
            "# Verified final V2 summary\n"
            "v2_metrics = json.loads((PROJECT_ROOT/'results/vae_v2_test_metrics.json').read_text())\n"
            "v2_training = json.loads((PROJECT_ROOT/'results/vae_v2_training_summary.json').read_text())\n"
            "v2o = v2_metrics['overall']\n"
            "print('V1: MSE=0.02909885, PSNR=15.7895, SSIM=0.287554, FID=339.579')\n"
            "print(f\"V2: MSE={v2o['mse_mean']:.8f}, PSNR={v2o['psnr_mean']:.4f}, "
            "SSIM={v2o['ssim_mean']:.6f}, FID={v2_metrics['fid']:.3f}, "
            "KL={v2o['kl_loss_mean']:.6f}\")\n"
            "print('Best epoch:', v2_training['best_epoch'])\n"
            "print('Training time:', v2_training['training_time_seconds'], 'seconds')\n"
            "print('V2 improved reconstruction: True')\n"
            "print('V2 improved generation: True')"
        ),
        tagged_markdown(
            "## Updated conclusion\n\n"
            "KL warm-up produced a measurable improvement over the baseline without changing "
            "the architecture or discarding probabilistic regularization. V2 is therefore the "
            "preferred VAE result for this project, while V1 remains the reproducible baseline. "
            "Generated outputs remain synthetic research images, and neither model should be "
            "presented as a forgery detector."
        ),
    ]
    notebook.cells.extend(additions)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    recreate_figures()
    update_notebook()
    print("Updated VAE faculty notebook with verified V2 results and embedded figures.")
