"""Build the authoritative same-split image reconstruction comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_TEST_IMAGES = 1892


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required verified result file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison(results_dir: Path) -> list[dict[str, object]]:
    """Read measured artifacts and normalize only compatible test metrics."""
    ae = read_json(results_dir / "quality_ae_v1_test_metrics.json")
    vae = read_json(results_dir / "vae_v5_final_test_metrics.json")
    transformer = read_json(results_dir / "transformer_v2_test_metrics.json")
    ae_architecture = read_json(results_dir / "quality_ae_v1_training_summary.json")
    vae_architecture = read_json(results_dir / "vae_v5_architecture_summary.json")

    rows = [
        {
            "model": "Quality Autoencoder V1",
            "primary_role": "Deterministic reconstruction and compression",
            "test_images": int(ae["number_of_test_images"]),
            "mse": float(ae["overall"]["mse_mean"]),
            "psnr_db": float(ae["overall"]["psnr_mean"]),
            "ssim": float(ae["overall"]["ssim_mean"]),
            "parameters": int(ae_architecture["parameter_count"]),
            "checkpoint_epoch": int(ae["checkpoint_epoch"]),
            "evaluation_seconds": float(ae["evaluation_time_seconds"]),
            "source_result": "results/quality_ae_v1_test_metrics.json",
        },
        {
            "model": "VAE V5 Final",
            "primary_role": "Probabilistic latent representation and reconstruction",
            "test_images": int(vae["test_images"]),
            "mse": float(vae["overall"]["mse_mean"]),
            "psnr_db": float(vae["overall"]["psnr_mean"]),
            "ssim": float(vae["overall"]["ssim_mean"]),
            "parameters": int(vae_architecture["total_parameters"]),
            "checkpoint_epoch": int(vae["checkpoint_metadata"]["epoch"]),
            "evaluation_seconds": float(vae["evaluation_seconds"]),
            "source_result": "results/vae_v5_final_test_metrics.json",
        },
        {
            "model": "Vision Transformer V2",
            "primary_role": "Patch-token self-attention reconstruction",
            "test_images": int(transformer["test_images"]),
            "mse": float(transformer["overall"]["mse_mean"]),
            "psnr_db": float(transformer["overall"]["psnr_mean"]),
            "ssim": float(transformer["overall"]["ssim_mean"]),
            "parameters": int(transformer["parameter_count"]),
            "checkpoint_epoch": int(transformer["checkpoint_metadata"]["epoch"]),
            "evaluation_seconds": float(transformer["evaluation_seconds"]),
            "source_result": "results/transformer_v2_test_metrics.json",
        },
    ]
    counts = {int(row["test_images"]) for row in rows}
    if counts != {EXPECTED_TEST_IMAGES}:
        raise ValueError(f"Comparison requires the same {EXPECTED_TEST_IMAGES}-image test split: {counts}")
    return rows


def write_comparison(rows: list[dict[str, object]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison_scope": "Same canonical held-out CASIA test split",
        "test_images_per_model": EXPECTED_TEST_IMAGES,
        "metric_interpretation": (
            "MSE, PSNR, and SSIM measure reconstruction fidelity; they are not tampering verdicts."
        ),
        "models": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--csv-path", type=Path, default=Path("results/image_model_comparison.csv")
    )
    parser.add_argument(
        "--json-path", type=Path, default=Path("results/image_model_comparison.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    comparison = build_comparison(arguments.results_dir)
    write_comparison(comparison, arguments.csv_path, arguments.json_path)
    print(json.dumps(comparison, indent=2))
