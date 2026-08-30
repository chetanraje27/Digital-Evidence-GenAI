# Digital Evidence — Autoencoder Component

Semester 7 project workspace for the convolutional Autoencoder (AE) component
of a multi-model digital-evidence analysis framework. VAE, GAN, Transformer,
and Diffusion implementations are outside this module's scope.

## Dataset validation

CASIA v2.0 is downloaded with KaggleHub into `data/raw`. After downloading,
run the reproducible dataset audit from the repository root:

```powershell
.\.venv\Scripts\python.exe src\explore_dataset.py --dataset-root data\raw
```

The audit checks every supported image for readability and records its label,
extension, dimensions, and color mode. It writes a JSON summary and CSV
inventories to `results/`, plus an authentic/tampered sample grid to `outputs/`.
Labels are preserved for experimental comparisons; reconstruction error alone
must not be interpreted as proof that an image is forged.
