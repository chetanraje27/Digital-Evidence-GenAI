from argparse import Namespace
from pathlib import Path

import torch

from train_autoencoder_v2 import checkpoint_config


def test_checkpoint_config_is_weights_only_safe(tmp_path: Path) -> None:
    args = Namespace(
        checkpoint_path=tmp_path / "model.pth",
        nested={"history": tmp_path / "history.csv"},
        values=[tmp_path / "grid.png", 42, True, None],
    )
    payload = {
        "model_state_dict": {"weight": torch.ones(1)},
        "config": checkpoint_config(args),
    }
    checkpoint_path = tmp_path / "safe_checkpoint.pth"
    torch.save(payload, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert loaded["config"]["checkpoint_path"] == str(args.checkpoint_path)
    assert loaded["config"]["nested"]["history"] == str(tmp_path / "history.csv")
    assert loaded["config"]["values"][0] == str(tmp_path / "grid.png")
