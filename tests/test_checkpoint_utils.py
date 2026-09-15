from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.checkpoint_utils import CheckpointUnavailableError, validate_checkpoint_file


class CheckpointValidationTests(unittest.TestCase):
    def test_real_bytes_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"PK\x03\x04model bytes")
            self.assertEqual(validate_checkpoint_file(path), path)

    def test_lfs_pointer_reports_expected_size_and_command(self) -> None:
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:abc\nsize 211297751\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "VAE_V5_FINAL.pth"
            path.write_bytes(pointer)
            with self.assertRaisesRegex(CheckpointUnavailableError, "211,297,751 expected bytes"):
                validate_checkpoint_file(path, "VAE V5")

    def test_missing_file_is_explicit(self) -> None:
        with self.assertRaises(FileNotFoundError):
            validate_checkpoint_file("does-not-exist.pth")


if __name__ == "__main__":
    unittest.main()
