from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_autoencoder_v2 import checkpoint_config


class AutoencoderCheckpointTests(unittest.TestCase):
    def test_checkpoint_config_is_weights_only_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory)
            args = Namespace(
                checkpoint_path=temporary_path / "model.pth",
                nested={"history": temporary_path / "history.csv"},
                values=[temporary_path / "grid.png", 42, True, None],
            )
            payload = {
                "model_state_dict": {"weight": torch.ones(1)},
                "config": checkpoint_config(args),
            }
            checkpoint_path = temporary_path / "safe_checkpoint.pth"
            torch.save(payload, checkpoint_path)

            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            self.assertEqual(loaded["config"]["checkpoint_path"], str(args.checkpoint_path))
            self.assertEqual(
                loaded["config"]["nested"]["history"], str(temporary_path / "history.csv")
            )
            self.assertEqual(loaded["config"]["values"][0], str(temporary_path / "grid.png"))


if __name__ == "__main__":
    unittest.main()
