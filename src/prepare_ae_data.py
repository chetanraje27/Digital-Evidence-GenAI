"""Create, validate, and visualize reproducible CASIA splits for the AE."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_dataset import create_ae_dataloaders


LABEL_MAP = {"authentic": 0, "tampered": 1}
EXPECTED_TOTAL = 12_614


def load_valid_inventory(inventory_path: Path, summary_path: Path) -> list[dict[str, str]]:
    """Reuse Stage 1 records and confirm their counts before splitting."""
    with summary_path.open(encoding="utf-8") as file:
        exploration_summary = json.load(file)
    with inventory_path.open(newline="", encoding="utf-8") as file:
        all_records = list(csv.DictReader(file))

    records = [
        row for row in all_records
        if row["label"] in LABEL_MAP and not row["error"]
    ]
    observed = Counter(row["label"] for row in records)
    expected = {
        name: exploration_summary["label_counts"][name] for name in LABEL_MAP
    }
    assert dict(observed) == expected, f"Inventory/summary count mismatch: {observed} != {expected}"
    assert len(records) == EXPECTED_TOTAL
    assert all(row["extension"].casefold() != ".png" for row in records)
    return records


def stratified_splits(
    records: list[dict[str, str]], seed: int
) -> dict[str, list[dict[str, str]]]:
    """Shuffle and allocate 70/15/15 independently inside each class."""
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, str]]] = {
        "train": [], "validation": [], "test": []
    }
    for class_name in LABEL_MAP:
        class_records = [row for row in records if row["label"] == class_name]
        rng.shuffle(class_records)
        train_end = round(len(class_records) * 0.70)
        validation_end = train_end + round(len(class_records) * 0.15)
        splits["train"].extend(class_records[:train_end])
        splits["validation"].extend(class_records[train_end:validation_end])
        splits["test"].extend(class_records[validation_end:])
    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def write_manifests(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        rows = [
            {
                "image_path": (Path("data/raw") / row["relative_path"]).as_posix(),
                "label": LABEL_MAP[row["label"]],
                "class_name": row["label"],
            }
            for row in records
        ]
        rows.sort(key=lambda row: row["image_path"])
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image_path", "label", "class_name"])
            writer.writeheader()
            writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_manifests(splits_dir: Path) -> tuple[dict[str, list[dict[str, str]]], int, int]:
    manifests = {
        name: read_manifest(splits_dir / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    path_sets = {
        name: {row["image_path"].casefold() for row in rows}
        for name, rows in manifests.items()
    }
    duplicate_count = sum(
        len(path_sets[left] & path_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    all_rows = [row for rows in manifests.values() for row in rows]
    mask_leakage_count = sum(
        Path(row["image_path"]).suffix.casefold() == ".png"
        or "groundtruth" in row["image_path"].casefold()
        for row in all_rows
    )
    assert len(all_rows) == EXPECTED_TOTAL
    assert len({row["image_path"].casefold() for row in all_rows}) == EXPECTED_TOTAL
    assert duplicate_count == 0
    assert mask_leakage_count == 0
    assert all(row["class_name"] in LABEL_MAP for row in all_rows)
    assert all(int(row["label"]) == LABEL_MAP[row["class_name"]] for row in all_rows)
    return manifests, duplicate_count, mask_leakage_count


def save_training_samples(manifest: list[dict[str, str]], project_root: Path, output_path: Path) -> None:
    """Save deterministic authentic and tampered examples from the training split."""
    groups = {
        class_name: [row for row in manifest if row["class_name"] == class_name][:5]
        for class_name in LABEL_MAP
    }
    figure, axes = plt.subplots(2, 5, figsize=(15, 6), squeeze=False)
    for row_index, class_name in enumerate(LABEL_MAP):
        for column_index, record in enumerate(groups[class_name]):
            with Image.open(project_root / record["image_path"]) as image:
                axes[row_index][column_index].imshow(image.convert("RGB"))
            axes[row_index][column_index].set_title(class_name.title())
            axes[row_index][column_index].axis("off")
    figure.suptitle("CASIA Autoencoder Training Samples")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["class_name"] for row in rows)
    return {name: counts[name] for name in LABEL_MAP}


def main(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    records = load_valid_inventory(args.inventory, args.exploration_summary)
    write_manifests(stratified_splits(records, args.seed), args.splits_dir)
    manifests, duplicate_count, mask_leakage_count = validate_manifests(args.splits_dir)

    loaders = create_ae_dataloaders(
        args.splits_dir, args.image_size, args.batch_size, args.num_workers, args.seed
    )
    batch = next(iter(loaders["train"]))
    images = batch["image"]
    expected_shape = (args.batch_size, 3, args.image_size, args.image_size)
    assert tuple(images.shape) == expected_shape, f"Unexpected batch shape: {tuple(images.shape)}"
    assert images.dtype == __import__("torch").float32
    pixel_min, pixel_max = float(images.min()), float(images.max())
    assert 0.0 <= pixel_min <= pixel_max <= 1.0

    save_training_samples(
        manifests["train"], project_root, args.outputs_dir / "training_samples.png"
    )
    summary: dict[str, object] = {
        "seed": args.seed,
        "split_percentages": {"train": 70, "validation": 15, "test": 15},
        "total_images": sum(len(rows) for rows in manifests.values()),
        "split_counts": {name: len(rows) for name, rows in manifests.items()},
        "class_counts": {name: class_counts(rows) for name, rows in manifests.items()},
        "class_ratios": {
            name: {
                class_name: round(count / len(rows), 6)
                for class_name, count in class_counts(rows).items()
            }
            for name, rows in manifests.items()
        },
        "batch_shape": list(images.shape),
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "duplicate_count_across_splits": duplicate_count,
        "ground_truth_mask_count_accidentally_included": mask_leakage_count,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "training_shuffle": True,
        "validation_shuffle": False,
        "test_shuffle": False,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("results/dataset_inventory.csv"))
    parser.add_argument("--exploration-summary", type=Path, default=Path("results/dataset_summary.json"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/ae"))
    parser.add_argument(
        "--output-summary", type=Path, default=Path("results/ae_data_pipeline_summary.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
    print(json.dumps(main(parse_args()), indent=2))
