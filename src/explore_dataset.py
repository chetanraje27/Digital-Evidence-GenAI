"""Validate and summarize the downloaded CASIA v2.0 image dataset.

This module performs dataset exploration only. It does not preprocess images,
create data splits, define a model, or train anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
AUTHENTIC_NAMES = {"au", "authentic", "original", "pristine", "real"}
TAMPERED_NAMES = {"tp", "tampered", "forged", "fake", "manipulated"}


def classify_image(path: Path) -> str:
    """Infer the class from actual path components, without fixed CASIA paths."""
    components = {part.casefold() for part in path.parts}
    if components & AUTHENTIC_NAMES:
        return "authentic"
    if components & TAMPERED_NAMES:
        return "tampered"
    return "unknown"


def inspect_image(path: Path) -> tuple[int, int, str]:
    """Fully verify an image, then reopen it to collect metadata."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        return image.width, image.height, image.mode


def save_sample_grid(records: list[dict[str, object]], output_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    groups = {
        label: [row for row in records if row["label"] == label and not row["error"]]
        for label in ("authentic", "tampered")
    }
    sample_count = min(5, *(len(group) for group in groups.values()))
    if sample_count == 0:
        print("Sample grid skipped: both authentic and tampered readable images are required.")
        return

    figure, axes = plt.subplots(2, sample_count, figsize=(3 * sample_count, 6), squeeze=False)
    for row_index, label in enumerate(("authentic", "tampered")):
        for column_index, record in enumerate(rng.sample(groups[label], sample_count)):
            with Image.open(str(record["absolute_path"])) as image:
                axes[row_index][column_index].imshow(image.convert("RGB"))
            axes[row_index][column_index].set_title(f"{label.title()}\n{record['filename']}", fontsize=8)
            axes[row_index][column_index].axis("off")
    figure.suptitle("CASIA v2.0 Dataset Samples")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def explore(dataset_root: Path, results_dir: Path, outputs_dir: Path, seed: int) -> dict[str, object]:
    dataset_root = dataset_root.resolve()
    image_paths = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported image files found under {dataset_root}")

    records: list[dict[str, object]] = []
    extensions: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    directory_counts: Counter[str] = Counter()

    for index, path in enumerate(image_paths, start=1):
        relative_path = path.relative_to(dataset_root)
        label = classify_image(relative_path)
        error = ""
        width = height = None
        mode = ""
        try:
            width, height, mode = inspect_image(path)
            dimensions[f"{width}x{height}"] += 1
            modes[mode] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        extension = path.suffix.casefold()
        extensions[extension] += 1
        labels[label] += 1
        directory_counts[str(relative_path.parent)] += 1
        records.append({
            "relative_path": relative_path.as_posix(),
            "absolute_path": str(path),
            "filename": path.name,
            "label": label,
            "extension": extension,
            "width": width,
            "height": height,
            "mode": mode,
            "error": error,
        })
        if index % 2000 == 0:
            print(f"Validated {index}/{len(image_paths)} images...")

    corrupt_records = [row for row in records if row["error"]]
    results_dir.mkdir(parents=True, exist_ok=True)
    inventory_fields = [
        "relative_path", "filename", "label", "extension", "width", "height", "mode", "error"
    ]
    with (results_dir / "dataset_inventory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=inventory_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (results_dir / "corrupt_images.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "label", "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(corrupt_records)

    summary: dict[str, object] = {
        "dataset": "CASIA v2.0 Image Tampering Detection Dataset",
        "kaggle_handle": "divg07/casia-20-image-tampering-detection-dataset",
        "dataset_root": str(dataset_root),
        "total_supported_image_files": len(image_paths),
        "label_counts": dict(sorted(labels.items())),
        "extension_counts": dict(sorted(extensions.items())),
        "color_mode_counts": dict(modes.most_common()),
        "common_dimensions": dict(dimensions.most_common(20)),
        "corrupt_or_unreadable_count": len(corrupt_records),
        "image_counts_by_directory": dict(directory_counts.most_common()),
        "label_inference": {
            "authentic_path_tokens": sorted(AUTHENTIC_NAMES),
            "tampered_path_tokens": sorted(TAMPERED_NAMES),
            "note": "Labels are inferred from discovered path component names and retained per image.",
        },
        "limitations": "Class labels support experimental comparison; they do not make reconstruction error proof of forgery.",
    }
    with (results_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_sample_grid(records, outputs_dir / "casia_sample_grid.png", seed)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = explore(arguments.dataset_root, arguments.results_dir, arguments.outputs_dir, arguments.seed)
    print(json.dumps(report, indent=2))
