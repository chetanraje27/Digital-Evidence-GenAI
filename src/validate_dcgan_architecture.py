"""Forward-pass smoke test for the standalone CASIA DCGAN architecture."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dcgan import build_dcgan


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate(
    batch_size: int = 32,
    latent_dim: int = 100,
    summary_path: Path = Path("results/gan_architecture_summary.json"),
) -> dict[str, object]:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator = build_dcgan(latent_dim=latent_dim)
    generator, discriminator = generator.to(device), discriminator.to(device)
    generator.eval()
    discriminator.eval()

    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        generated = generator(noise)
        logits = discriminator(generated)

    expected_generated_shape = (batch_size, 3, 64, 64)
    expected_logit_shape = (batch_size, 1, 1, 1)
    finite = bool(torch.isfinite(generated).all() and torch.isfinite(logits).all())
    generated_min = float(generated.min())
    generated_max = float(generated.max())

    assert tuple(generated.shape) == expected_generated_shape
    assert tuple(logits.shape) == expected_logit_shape
    assert generated_min >= -1.0 and generated_max <= 1.0
    assert finite

    summary: dict[str, object] = {
        "device": str(device),
        "latent_dim": latent_dim,
        "noise_shape": list(noise.shape),
        "generated_shape": list(generated.shape),
        "generated_min": generated_min,
        "generated_max": generated_max,
        "discriminator_output_shape": list(logits.shape),
        "discriminator_output_type": "logits",
        "generator_parameters": parameter_count(generator),
        "discriminator_parameters": parameter_count(discriminator),
        "nan_status": not finite,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = validate()
    print("Generator parameters:", result["generator_parameters"])
    print("Discriminator parameters:", result["discriminator_parameters"])
    print("Generated shape:", result["generated_shape"])
    print(
        "Generated range:",
        f"[{result['generated_min']:.6f}, {result['generated_max']:.6f}]",
    )
    print("Discriminator output shape:", result["discriminator_output_shape"])
    print("NaN status:", result["nan_status"])
