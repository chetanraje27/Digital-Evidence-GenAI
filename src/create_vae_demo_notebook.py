"""Build the single, load-only VAE faculty demonstration notebook."""

from __future__ import annotations

import csv
import base64
import json
import re
from pathlib import Path

import matplotlib
import nbformat as nbf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRAINING_NOTEBOOK = ROOT / "notebooks" / "03_vae_training_colab.ipynb"
HISTORY_PATH = ROOT / "results" / "vae_training_history.csv"
TOTAL_CURVE = ROOT / "outputs" / "vae" / "vae_total_loss_curve.png"
COMPONENT_CURVE = ROOT / "outputs" / "vae" / "vae_reconstruction_kl_curve.png"
COMPARISON_PLOT = ROOT / "outputs" / "vae" / "vae_authentic_vs_tampered_metrics.png"


def recover_history() -> list[dict[str, str | int | float]]:
    notebook = json.loads(TRAINING_NOTEBOOK.read_text(encoding="utf-8"))
    output = "\n".join(
        "".join(item.get("text", []))
        for cell in notebook["cells"] for item in cell.get("outputs", [])
        if item.get("output_type") == "stream"
    )
    pattern = re.compile(
        r"Epoch (\d+)/30 \| train_total=([\d.]+) \| val_total=([\d.]+) "
        r"\| recon=([\d.]+) \| kl=([\d.]+) \| seconds=([\d.]+)"
    )
    rows = []
    for match in pattern.finditer(output):
        epoch, train_total, val_total, val_recon, val_kl, seconds = match.groups()
        rows.append({
            "epoch": int(epoch), "train_total_loss": float(train_total),
            "train_reconstruction_loss": "", "train_kl_loss": "",
            "validation_total_loss": float(val_total),
            "validation_reconstruction_loss": float(val_recon),
            "validation_kl_loss": float(val_kl), "epoch_time_seconds": float(seconds),
        })
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 recorded epochs, found {len(rows)}")
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_training_plots(rows: list[dict[str, str | int | float]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    TOTAL_CURVE.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [float(row["train_total_loss"]) for row in rows], label="Train total")
    axis.plot(epochs, [float(row["validation_total_loss"]) for row in rows], label="Validation total")
    axis.set(title="VAE Total Loss — Recorded 30-Epoch GPU Run", xlabel="Epoch", ylabel="Total loss")
    axis.grid(alpha=.3); axis.legend(); figure.tight_layout(); figure.savefig(TOTAL_CURVE, dpi=160); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [float(row["validation_reconstruction_loss"]) for row in rows], color="tab:green")
    axes[0].set(title="Validation Reconstruction Loss", xlabel="Epoch", ylabel="MSE")
    axes[1].plot(epochs, [float(row["validation_kl_loss"]) for row in rows], color="tab:orange")
    axes[1].set(title="Validation KL Divergence", xlabel="Epoch", ylabel="Mean KL per sample")
    for axis in axes: axis.grid(alpha=.3)
    figure.suptitle("Recorded Validation Components (training components were not retained by Colab output)")
    figure.tight_layout(); figure.savefig(COMPONENT_CURVE, dpi=160); plt.close(figure)


def make_comparison_plot() -> None:
    metrics = json.loads((ROOT / "results" / "vae_test_metrics.json").read_text(encoding="utf-8"))
    names = ["MSE", "PSNR", "SSIM"]
    authentic = [metrics["authentic"][f"{name.lower()}_mean"] for name in names]
    tampered = [metrics["tampered"][f"{name.lower()}_mean"] for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for index, (axis, name) in enumerate(zip(axes, names)):
        bars = axis.bar(["Authentic", "Tampered"], [authentic[index], tampered[index]], color=["#4c78a8", "#f58518"])
        axis.set_title(f"Mean {name}"); axis.bar_label(bars, fmt="%.4f", padding=3); axis.grid(axis="y", alpha=.25)
    figure.suptitle("Exploratory reconstruction comparison — not a forgery classifier")
    figure.tight_layout(); figure.savefig(COMPARISON_PLOT, dpi=160, bbox_inches="tight"); plt.close(figure)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def attached_figure(title: str, path: Path):
    """Embed a figure so GitHub/Colab display it even though outputs/ is ignored."""
    name = path.name
    cell = markdown(f"### {title}\n\n![{title}](attachment:{name})")
    cell["attachments"] = {
        name: {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
    }
    return cell


def build_notebook() -> None:
    cells = [
        markdown("# VAE for CASIA Digital Evidence Analysis\n\n**Faculty demonstration — VAE component only**\n\nObjective: learn a probabilistic latent representation for image reconstruction and synthetic generation. Reconstruction behavior is explored for authentic and tampered images; it is **not** presented as proof of forgery."),
        markdown("## Section 1 — Project and VAE objective\n\nThis VAE module belongs to *Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation*. It reconstructs 128×128 RGB images, learns a continuous latent distribution, and generates synthetic research images."),
        code("""# Run once at the top. In Colab this clones code only when needed.\nfrom pathlib import Path\nimport os, sys, subprocess\nIN_COLAB = 'google.colab' in sys.modules\nrepo_name = 'Digital-Evidence-GenAI'\nif IN_COLAB:\n    base = Path('/content')\n    PROJECT_ROOT = base / repo_name\n    if not PROJECT_ROOT.exists():\n        subprocess.run(['git','clone','https://github.com/chetanraje27/Digital-Evidence-GenAI.git',str(PROJECT_ROOT)], check=True)\n    subprocess.run([sys.executable,'-m','pip','install','-q','kagglehub','scikit-image','torchmetrics','torch-fidelity'], check=True)\nelse:\n    here = Path.cwd().resolve()\n    PROJECT_ROOT = next((p for p in [here, *here.parents] if (p/'src'/'vae.py').exists()), here)\nsys.path.insert(0, str(PROJECT_ROOT/'src'))\nTRAIN_VAE = False  # Faculty demo default: never retrain\nprint('Project:', PROJECT_ROOT)\nprint('TRAIN_VAE:', TRAIN_VAE)"""),
        markdown("## Section 2 — CASIA dataset overview\n\n- CASIA v2.0: **12,614** RGB images\n- Authentic: **7,491**; tampered: **5,123**\n- Ground-truth masks are excluded from VAE input\n- Held-out test set: **1,892** images (1,123 authentic, 769 tampered)"),
        markdown("## Section 3 — Data preprocessing and splits\n\nImages are safely opened with PIL, converted to RGB, resized to 128×128, converted to tensors, and scaled to `[0,1]` without ImageNet normalization. Reproducible seed-42 stratified splits are 70% train, 15% validation, and 15% test. Labels are metadata only."),
        code("""# Download CASIA only if the relative manifest images are unavailable (Colab).\nimport csv\nfirst = next(csv.DictReader(open(PROJECT_ROOT/'data/splits/test.csv', encoding='utf-8')))\nexpected = PROJECT_ROOT / first['image_path']\nif not expected.exists():\n    import kagglehub\n    downloaded = Path(kagglehub.dataset_download('divg07/casia-20-image-tampering-detection-dataset'))\n    casia2 = next(downloaded.rglob('CASIA2'))\n    target = PROJECT_ROOT/'data/raw/CASIA2'\n    target.parent.mkdir(parents=True, exist_ok=True)\n    if not target.exists(): os.symlink(casia2, target, target_is_directory=True)\nprint('Dataset ready:', (PROJECT_ROOT/'data/raw/CASIA2').exists())"""),
        markdown("## Section 4 — VAE architecture\n\n**Encoder:** `3×128×128 → 32×64×64 → 64×32×32 → 128×16×16 → 128×8×8`\n\nThe flattened features feed separate **μ** and **log-variance** heads. Reparameterization produces latent **z**, and the transposed-convolution decoder reconstructs `3×128×128` with sigmoid output.\n\n- Latent dimension: **128**\n- Parameters: **4,009,795**"),
        code("""import torch\nfrom vae import ConvolutionalVAE\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel = ConvolutionalVAE(latent_dim=128).to(device)\nprint(model)\nprint('Parameters:', sum(p.numel() for p in model.parameters()))"""),
        markdown("## Section 5 — Mathematical idea\n\nThe encoder learns a Gaussian distribution and samples differentiably:\n\n$$z = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0,I)$$\n\n$$\\mathcal{L}_{total}=\\mathcal{L}_{reconstruction}+\\beta D_{KL}, \\qquad \\beta=0.001$$\n\nMSE encourages reconstruction fidelity; KL regularization encourages a smooth latent distribution."),
        markdown("## Section 6 — Training configuration\n\n| Setting | Value |\n|---|---:|\n| Batch size | 32 |\n| Learning rate | 0.0005 |\n| Epochs | 30 |\n| Optimizer | Adam |\n| Latent dimension | 128 |\n| β | 0.001 |"),
        markdown("## Section 7 — Training results\n\n- GPU: Tesla T4\n- Best epoch: **30**\n- Best validation total loss: **0.04149372**\n- Training time: **1554.43 seconds**\n\nThe following are genuine curves recovered from the recorded Colab run. Per-epoch training reconstruction/KL components were not retained, so component plots show validation values only."),
        attached_figure("Total training and validation loss", TOTAL_CURVE),
        attached_figure("Validation reconstruction and KL curves", COMPONENT_CURVE),
        markdown("## Section 8 — Reconstruction results"),
        attached_figure("Original and reconstructed test images", ROOT / "outputs/vae/vae_test_reconstruction_grid.png"),
        markdown("## Section 9 — Test metrics\n\nMetrics are computed per image on the complete held-out test split using deterministic `z = μ`, then averaged. KL is the mean per-sample latent KL; total loss is MSE + 0.001 × KL."),
        code("""import json\nmetrics_path = PROJECT_ROOT/'results/vae_test_metrics.json'\nmetrics = json.loads(metrics_path.read_text())\no = metrics['overall']\nprint(f\"Test images: {metrics['test_images']}\")\nprint(f\"MSE: {o['mse_mean']:.8f} | PSNR: {o['psnr_mean']:.4f} dB | SSIM: {o['ssim_mean']:.4f}\")\nprint(f\"KL: {o['kl_loss_mean']:.6f} | Total loss: {o['total_loss_mean']:.8f}\")"""),
        markdown("## Section 10 — Synthetic generation\n\nSamples below are decoded from `z ~ N(0,I)`. They are **synthetic VAE-generated research outputs, not real forensic evidence**."),
        attached_figure("Synthetic VAE-generated samples", ROOT / "outputs/vae/vae_generated_samples.png"),
        markdown("## Section 11 — Latent interpolation\n\nLinear interpolation between two encoded test-image means illustrates a continuous learned latent space."),
        attached_figure("Latent interpolation", ROOT / "outputs/vae/vae_latent_interpolation.png"),
        markdown("## Section 12 — FID\n\nLower FID means the generated-image feature distribution is closer to the real-image distribution. This standard Inception-V3 FID uses **1,892 real test images and 1,892 separately generated images**—not only the 25 display samples."),
        code("print(f\"FID: {metrics['fid']:.4f} ({metrics['fid_implementation']})\")"),
        markdown("## Section 13 — Authentic vs tampered exploratory comparison\n\nThese statistics describe reconstruction behavior only. They do **not** make the VAE a forgery detector and do not prove whether an image is forged."),
        attached_figure("Authentic vs tampered exploratory metrics", COMPARISON_PLOT),
        markdown("## Section 14 — Conclusion and limitations\n\n- The VAE learned probabilistic latent representations for reconstruction and generation.\n- Generated images are synthetic research outputs.\n- VAE samples may look smoother or blurrier than GAN outputs.\n- Reconstruction statistics are exploratory, not forensic verdicts.\n- The team can later compare VAE behavior with AE and GAN results."),
        markdown("## Optional live inference (no training)"),
        code("""# Loads the trained checkpoint; it does not retrain.\ncheckpoint_path = PROJECT_ROOT/'checkpoints/best_vae.pth'\ncheckpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)\nmodel.load_state_dict(checkpoint['model_state_dict']); model.eval()\nprint('Loaded:', checkpoint_path, '| epoch:', checkpoint['epoch'])"""),
        code("""# Final faculty-demo summary\na, t = metrics['authentic'], metrics['tampered']\nprint(f\"Test images: {metrics['test_images']}\")\nprint(f\"Test reconstruction MSE: {o['mse_mean']:.8f}\")\nprint(f\"Test PSNR: {o['psnr_mean']:.6f}\")\nprint(f\"Test SSIM: {o['ssim_mean']:.6f}\")\nprint(f\"Test KL: {o['kl_loss_mean']:.6f}\")\nprint(f\"Test total loss: {o['total_loss_mean']:.8f}\")\nprint(f\"FID: {metrics['fid']:.6f}\")\nprint(f\"Authentic MSE/PSNR/SSIM: {a['mse_mean']:.8f} / {a['psnr_mean']:.6f} / {a['ssim_mean']:.6f}\")\nprint(f\"Tampered MSE/PSNR/SSIM: {t['mse_mean']:.8f} / {t['psnr_mean']:.6f} / {t['ssim_mean']:.6f}\")\nprint('Generated samples: 25 display; 1,892 for FID')\nprint('Latent dimension: 128')\nprint('Checkpoint:', checkpoint_path)"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "VAE_CASIA_Complete_Demo.ipynb", "provenance": []},
    })
    output = ROOT / "notebooks" / "VAE_CASIA_Complete_Demo.ipynb"
    nbf.write(notebook, output)


if __name__ == "__main__":
    recovered = recover_history()
    make_training_plots(recovered)
    make_comparison_plot()
    build_notebook()
    print("Created VAE history, plots, comparison, and faculty notebook.")
