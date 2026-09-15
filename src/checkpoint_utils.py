"""Checkpoint availability checks with explicit Git LFS pointer diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class CheckpointUnavailableError(RuntimeError):
    """Raised when a checkpoint path contains metadata but not model bytes."""


def validate_checkpoint_file(path: str | Path, label: str = "Model") -> Path:
    """Return an existing checkpoint path or raise an actionable exception."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        prefix = file.read(512)
    if prefix.startswith(LFS_HEADER):
        pointer = prefix.decode("utf-8", errors="replace")
        expected_size = re.search(r"^size (\d+)$", pointer, flags=re.MULTILINE)
        size_note = f" ({int(expected_size.group(1)):,} expected bytes)" if expected_size else ""
        fetch_path = checkpoint_path.as_posix()
        if checkpoint_path.is_absolute():
            try:
                fetch_path = checkpoint_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:
                pass
        raise CheckpointUnavailableError(
            f"{label} checkpoint is a Git LFS pointer, not model bytes{size_note}: "
            f"{checkpoint_path}. Run 'git lfs install' and "
            f"'git lfs pull --include=\"{fetch_path}\"', then retry."
        )
    return checkpoint_path


__all__ = ["CheckpointUnavailableError", "validate_checkpoint_file"]
