# AE RTX-80 Training Transfer Instructions

## Copy to the RTX laptop

The simplest and safest option is to copy the complete `Digital_Evidence` project folder. At minimum, the copied project must contain:

- `notebooks/AE/03_autoencoder_training_rtx_80epochs.ipynb`
- `src/autoencoder.py`
- `src/ae_dataset.py`
- `src/train_autoencoder_v2.py`
- `src/evaluate_autoencoder.py`
- `data/splits/train.csv`
- `data/splits/validation.csv`
- `data/splits/test.csv`
- `data/raw/CASIA2/Au/`
- `data/raw/CASIA2/Tp/`
- `requirements.txt`

Ground-truth mask folders are not required for AE training. Do not place masks inside `Au` or `Tp`.

## RTX laptop preparation

1. Install a current NVIDIA driver.
2. Install Python and Jupyter or VS Code.
3. Install the CUDA-enabled PyTorch build from the official PyTorch installation selector.
4. Open the copied repository in VS Code.
5. Open `notebooks/AE/03_autoencoder_training_rtx_80epochs.ipynb`.
6. Select the environment containing CUDA-enabled PyTorch.
7. Run all cells.

The notebook refuses to train if `torch.cuda.is_available()` is false.

## Copy back after training

Copy this generated file back to the original laptop:

```text
AE_RTX80_RETURN_ARTIFACTS.zip
```

It contains the best checkpoint, training history, summary, complete test metrics and visual outputs. Extract it into the original repository root while preserving its folder structure.

The previous checkpoint is not overwritten. The new checkpoint is:

```text
checkpoints/best_autoencoder_rtx80.pth
```
