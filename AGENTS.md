# AGENTS.md

## Project
Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation

This repository is a Semester 7 Generative AI academic project. The current Review-2 priority is to make the implementation technically correct, demonstrable, reproducible, and aligned with faculty feedback.

Before changing code, read `REVIEW2_CONTEXT.md` completely.

---

## Core working rules

1. Inspect existing code, checkpoints, result files, and data manifests before editing.
2. Preserve working functionality unless a change is explicitly required.
3. Never delete or overwrite a working checkpoint without creating/preserving the old version.
4. Never fabricate, estimate, or hard-code evaluation results as if they were measured.
5. Only display a metric as an actual result if it was produced by a real evaluation run or already exists in a verified repository result file.
6. Keep model claims conservative:
   - reconstruction error is not a tampering probability;
   - attention is not proof of manipulation;
   - AE/VAE/Transformer are not standalone forensic verdict systems.
7. Keep the current Vision Transformer implementation. It is technically a valid Transformer model.
8. Add the faculty-requested prompt/text/LLM Transformer as a separate module. Do not pretend the Vision Transformer is not a Transformer.
9. Keep GAN code/checkpoints preserved for the later 60-mark phase, but do not put GAN back into the current Review-2 GUI unless explicitly requested.
10. Diffusion is not implemented yet. Do not claim otherwise.
11. Do not update the final PPT-facing documentation until implementation and results are verified.
12. Avoid unnecessary architecture rewrites. Prefer the smallest technically correct change.
13. Run tests after every meaningful change.
14. If GPU training is required and cannot be performed locally, prepare a complete runnable script/notebook, exact command, expected artifacts, and stop only for that genuine blocker.
15. If a checkpoint is a Git LFS pointer instead of actual model bytes, detect it and report it clearly rather than failing mysteriously.
16. Keep Windows and Linux/Streamlit deployment portability in mind.
17. Do not introduce API keys or paid external services as mandatory dependencies.
18. Prefer a no-key local/open-source Transformer/LLM path for the faculty demo.
19. Avoid misleading GDPR claims. Implement privacy-oriented controls and describe them as design measures/principles unless formal compliance has been established.
20. Keep the UI professional, consistent, and suitable for a live faculty demo.

---

## Scientific rules

### Autoencoder
- Primary role: deterministic reconstruction/compression.
- Use the improved RTX-80 checkpoint as the current final AE.
- Do not present AE as a forgery classifier.

### VAE
- Primary role: probabilistic latent representation plus reconstruction.
- Show actual stochastic behavior using the learned `mu`, `logvar`, reparameterization, and multiple image-conditioned latent samples.
- Do not fake "new features" by manually editing pixels.
- If current skip connections make stochastic outputs nearly identical, measure that honestly and prepare a principled retraining option rather than fabricating visible differences.

### Vision Transformer
- Primary role: patch-token self-attention reconstruction and exploratory attention visualization.
- Preserve the current image Transformer.
- Evaluate it on the same canonical held-out test split used by AE/VAE before direct reconstruction comparison.

### Prompt/Text Transformer
- This is an additional faculty-facing module.
- It should accept text/PDF evidence, extract text, tokenize it, accept a user prompt, and generate evidence-oriented output with a pretrained Transformer language model.
- It must be clearly labelled as a pretrained language model integration, not a model trained from scratch by the team.
- Generated output must include a limitation/responsible-use note.

---

## Coding conventions

- Python 3.10+ compatible.
- Keep modules under `src/`.
- Use `pathlib.Path`.
- Use type hints for public functions where practical.
- Keep model loading cached in Streamlit.
- Gracefully handle missing checkpoints, bad uploads, empty PDFs, unsupported files, model-download failures, and CPU-only environments.
- Keep deterministic seeds for evaluation where applicable.
- Put generated evaluation artifacts in `results/` and selected presentation-ready images in `docs/results/`.
- Bulk generated artifacts may remain ignored.
- Prefer functions/modules over large monolithic additions to `app.py`.
- Do not silently catch errors; display actionable messages in the GUI and useful exceptions in scripts.

---

## Validation before declaring a phase complete

For each phase:
1. Run syntax/import checks.
2. Run relevant unit/smoke tests.
3. Run at least one real inference path if model bytes are available.
4. Verify the GUI does not crash when the corresponding page is opened.
5. Verify labels, captions, and numbers match the actual implementation.
6. Record exactly what changed.
7. Record what was tested.
8. Record any blocker that requires user/GPU/deployment action.

Do not say "complete" if a critical path was not actually tested.
