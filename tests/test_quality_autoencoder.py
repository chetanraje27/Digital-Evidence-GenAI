from __future__ import annotations

import unittest

import torch

from evaluate_autoencoder import build_model
from quality_autoencoder import ARCHITECTURE_NAME, QualityAutoencoder
from train_quality_autoencoder import QualityReconstructionLoss


class QualityAutoencoderTests(unittest.TestCase):
    def test_shape_bottleneck_and_no_transposed_convolution(self) -> None:
        model = QualityAutoencoder().eval()
        inputs = torch.rand(2, 3, 128, 128)
        with torch.inference_mode():
            latent = model.encode(inputs)
            outputs = model(inputs)
        self.assertEqual(tuple(latent.shape), (2, 32, 16, 16))
        self.assertEqual(tuple(outputs.shape), tuple(inputs.shape))
        self.assertTrue(bool(torch.isfinite(outputs).all()))
        self.assertFalse(any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules()))

    def test_identity_has_near_zero_quality_loss(self) -> None:
        criterion = QualityReconstructionLoss(0.65, 0.25, 0.10)
        inputs = torch.rand(2, 3, 32, 32)
        self.assertLess(float(criterion(inputs, inputs)), 1e-6)

    def test_evaluator_selects_quality_architecture(self) -> None:
        model, architecture, compression_ratio = build_model({"architecture": ARCHITECTURE_NAME})
        self.assertIsInstance(model, QualityAutoencoder)
        self.assertEqual(architecture, ARCHITECTURE_NAME)
        self.assertEqual(compression_ratio, 6)

    def test_loss_weights_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            QualityReconstructionLoss(0.5, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
