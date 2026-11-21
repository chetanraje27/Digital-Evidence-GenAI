# REVIEW2_CONTEXT.md

# Review-2 Project Context and Implementation Plan

## 1. Project identity

**Title:** Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation

**Current course phase:** Semester 7 Generative AI, Review 2 preparation.

**Current required three-model scope for the 30-mark practical phase:**
1. Autoencoder
2. Variational Autoencoder
3. Transformer-based architecture

GAN is already implemented and must be preserved for the later 60-mark phase. Diffusion remains future work.

The purpose of this file is to give Codex all important project context in one place so implementation can continue in one persistent Codex chat without repeatedly re-explaining the project.

---

# 2. What happened in Review 1

The faculty member, Savita Mane, was not satisfied with the project being presented mostly as theory plus basic image reconstruction. The core feedback was:

## 2.1 Results must be visible
The PPT and GUI must show actual implementation results for every model, not only architecture/theory.

The faculty explicitly asked for:
- individual model results;
- evaluation metrics;
- a comparison table;
- professional model-specific GUI pages.

## 2.2 Autoencoder
Faculty observations:
- reconstructed output shown during Review 1 was not clear enough;
- observe reconstruction loss and improve reconstruction;
- prepare approximately 20–25 validation/testing samples for live demonstration;
- label outputs professionally as:
  - `Original Image`
  - `Reconstructed Image`
  rather than generic `Input` / `Model output`.

The team subsequently retrained the AE for 80 epochs and obtained a better checkpoint.

## 2.3 VAE
Faculty expects the VAE module to visibly demonstrate the probabilistic nature of VAE, not only display an almost-identical deterministic reconstruction.

The faculty's wording suggested "adding new features" and visibly different outputs through the probability distribution. The technically correct interpretation for this implementation is:

- keep the original evidence image;
- show deterministic VAE reconstruction using `z = mu`;
- show multiple image-conditioned stochastic reconstructions using:
  `z = mu + temperature * sigma * epsilon`;
- explain that these are probabilistic latent variations, not manually added features and not proof of manipulation.

If current VAE skip connections make all stochastic outputs nearly identical, do not fake the difference. Measure the behavior and, if needed, prepare a retrained VAE variant with more meaningful latent usage.

## 2.4 Transformer
The existing project Transformer is a valid image Transformer autoencoder using:
- image patches;
- patch embeddings;
- positional embeddings;
- multi-head self-attention;
- Transformer encoder/decoder blocks;
- image reconstruction.

However, the faculty specifically expects a prompt/text/LLM-style Transformer demonstration. The faculty mentioned:
- 5–8 line prompt;
- LLM/SLM;
- tokenization;
- PDF evidence;
- generated output.

Therefore Review 2 should contain **both**:

### A. Existing Vision Transformer
Preserve it as the trained Transformer architecture required by the course.

### B. New Prompt/Text Transformer Evidence Intelligence module
Add a separate pretrained Transformer language model integration:
`PDF/Text Evidence -> text extraction -> tokenization -> prompt -> Transformer language model -> generated evidence intelligence`

Do not replace or delete the current Vision Transformer.

## 2.5 Evaluation
Faculty wants evaluation parameters applied appropriately and a comparison table in both the GUI and PPT.

For image reconstruction models, direct comparison is only valid if they are evaluated on the same test split.

Required reconstruction metrics:
- MSE
- PSNR
- SSIM
- parameter count
- checkpoint/best epoch
- optionally inference/evaluation time

Do not force MSE/PSNR/SSIM onto generated text from the LLM module. The existing Vision Transformer provides the image-model Transformer comparison.

## 2.6 GUI
Faculty asked for a more professional GUI with clear model separation and result presentation.

## 2.7 Ethical deployment
Faculty asked specifically about:
- privacy;
- security;
- ethical deployment;
- GDPR-related rules/principles.

The app must implement practical privacy-oriented controls rather than merely saying Streamlit is secure.

## 2.8 Literature/PPT
Faculty requested approximately 12–15 literature sources with columns such as:
- Sr. No.
- Year
- Paper title
- Description
- Exact idea extracted for this project

PPT improvement comes after implementation/results are finalized.

---

# 3. Verified current repository state

The following facts were checked directly from the current GitHub ZIP.

## 3.1 Existing GUI

Current `app.py` pages:
1. Project Overview
2. Autoencoder
3. VAE V5 Final
4. Transformer V2
5. Model Comparison

Current app imports:
- `AutoencoderInference`
- `VAEInference`
- `TransformerInference`

No prompt/text LLM module currently exists.

---

# 4. Autoencoder: current verified final state

## 4.1 Architecture
Current AE:
- input: `3 x 128 x 128`
- latent: `32 x 8 x 8`
- latent values: 2,048
- input values: 49,152
- compression ratio: `24x`
- parameters: `265,571`
- output: reconstructed `3 x 128 x 128` RGB image

## 4.2 Improved RTX-80 run
Verified file:
`results/ae_rtx80_test_metrics.json`

Best checkpoint:
- checkpoint epoch: `76`
- validation loss: `0.0030500478`

Held-out test:
- test images: `1,892`
- authentic: `1,123`
- tampered: `769`

Overall:
- MSE: `0.0030319858`
- PSNR: `26.015853 dB`
- SSIM: `0.7929106`

The 80-epoch history shows the best validation loss around epoch 76.

## 4.3 Important current bug
`app.py` currently says it is loading the improved epoch-76 RTX-80 checkpoint, but `load_ae()` actually points to:

`checkpoints/best_autoencoder.pth`

The intended current checkpoint is:

`checkpoints/best_autoencoder_rtx80_portable.pth`

Fix the app to use the portable RTX-80 checkpoint.

Also check any evaluation script defaults that still point to the old checkpoint.

## 4.4 Current UI wording issue
The helper `compare_images()` still renders small headings:
- `Input`
- `Model output`

Faculty explicitly requested:
- `Original Image`
- `Reconstructed Image`

Fix the headings everywhere.

## 4.5 AE Review-2 acceptance criteria
AE phase is complete only when:
- the app loads the RTX-80 portable checkpoint;
- original/reconstructed labels are correct;
- verified test metrics match the RTX-80 result file;
- a real sample inference works;
- the page clearly states MSE loss and deterministic reconstruction role;
- no authenticity/tampering verdict is produced.

---

# 5. VAE V5: current verified state

## 5.1 Architecture
Current model class:
`src/vae.py -> VAEV5`

Verified properties:
- input: `3 x 128 x 128`
- encoder channels: 32, 64, 128, 256
- residual blocks
- GroupNorm
- SiLU
- latent dimension: `256`
- separate `mu` and `logvar`
- skip-connected decoder
- final Sigmoid
- parameters: `17,599,971`

## 5.2 Loss
Verified VAE V5 loss:
- MSE
- L1
- KL divergence

Reconstruction:
`(1 - l1_weight) * MSE + l1_weight * L1`

Current `l1_weight = 0.5`, therefore:
`0.5 * MSE + 0.5 * L1`

Total:
`reconstruction_loss + beta * KL`

KL beta warm-up:
- starts near `0.00005`
- reaches `0.00030`
- warm-up over first 20 epochs

## 5.3 Current final metrics
Verified file:
`results/vae_v5_final_test_metrics.json`

Checkpoint metadata:
- epoch: `79`
- val MSE: `0.0009786994`
- val PSNR: `30.093507 dB`
- val KL: `0.00867804`

Canonical held-out test:
- images: `1,892`
- MSE: `0.0007000721`
- PSNR: `32.953456 dB`
- SSIM: `0.96113885`
- mean KL: `0.01292955`

Exploratory anomaly ranking:
- MSE ROC-AUC: `0.5153818`
- SSIM-error ROC-AUC: `0.5705580`

Interpretation:
VAE V5 reconstructs very strongly, but reconstruction error is weak for authentic/tampered separation. Do not present it as a standalone forgery detector.

## 5.4 Current VAE GUI
Current GUI has:
1. deterministic reconstruction with `z = mu`;
2. random prior generation with `z ~ N(0,I)`.

The random prior generation is known to be weak/dark because the decoder relies heavily on encoder skip features.

## 5.5 Required VAE change for Review 2
Add **image-conditioned stochastic reconstructions**.

Implement an inference API such as:

`reconstruct_variations(image, n_samples=3, temperature=1.0, seed=None)`

Correct internal flow:
1. preprocess image;
2. run encoder to get:
   - `mu`
   - `logvar`
   - `e1,e2,e3,e4`
3. calculate:
   - `std = exp(0.5 * logvar)`
4. for each variation:
   - `epsilon ~ N(0,I)`
   - `z = mu + temperature * std * epsilon`
5. decode with the same image skip tensors;
6. return:
   - original image;
   - deterministic reconstruction;
   - multiple stochastic reconstructions;
   - metrics for deterministic reconstruction;
   - optional per-variation metrics/latent distance.

Suggested GUI:
- Original Image
- Deterministic Reconstruction (`z = mu`)
- Stochastic Variation 1
- Stochastic Variation 2
- Stochastic Variation 3
- temperature control in a safe range

Use wording:
`Probabilistic latent variations / stochastic reconstructions`

Do not call them guaranteed new forensic features.

## 5.6 If variations remain visually identical
Because skip connections are strong, VAE stochastic variation may be very weak.

If this happens:
1. quantify it;
2. try a sensible temperature range;
3. do not fake differences;
4. if still weak, prepare a **new VAE Review-2 training variant** using a principled design such as:
   - weaker/reduced skip connections;
   - skip dropout during training;
   - stronger but stable KL contribution;
   - latent sensitivity checks;
   - while preserving reconstruction quality reasonably.

Do not overwrite `VAE_V5_FINAL.pth`.

## 5.7 VAE checkpoint Git LFS issue
In the GitHub ZIP, `checkpoints/VAE_V5_FINAL.pth` is only a 134-byte Git LFS pointer.

It points to an actual file size of approximately 211 MB.

Therefore:
- normal GitHub ZIP download does not contain the model bytes;
- local/deployment loading may fail unless Git LFS fetches the model.

Implement clear checkpoint validation:
- detect tiny text Git LFS pointer files;
- show a useful error;
- do not pass them into `torch.load`.

For deployment, prepare a reproducible strategy such as:
- Git LFS fetch when supported, or
- model hosted as a release/model artifact and downloaded/cached at startup.

Do not commit duplicate 211 MB files without considering repository/deployment constraints.

---

# 6. Existing Vision Transformer V2: current verified state

## 6.1 Architecture
File:
`src/transformer.py`

Verified architecture:
- input: `3 x 128 x 128`
- patch size: `16 x 16`
- patch grid: `8 x 8`
- patch tokens: `64`
- embedding dimension: `256`
- attention heads: `8`
- encoder layers: `4`
- decoder layers: `2`
- feed-forward dimension: `512`
- learnable positional embeddings
- multi-head self-attention
- token reconstruction back to image
- final Sigmoid

This is a genuine Transformer-based image autoencoder.

## 6.2 Checkpoint
`checkpoints/transformer_v2_final.pth`

Verified metadata:
- epoch: `50`
- train MSE: `0.0025685047`
- validation MSE: `0.0020600450`
- anomaly threshold: `0.0044039055`
- ROC-AUC: `0.5455343`
- balanced accuracy: `0.5189541`

Current anomaly performance is weak/near-random and must be described honestly.

## 6.3 Current comparison problem
The GUI currently compares:
- AE canonical test MSE
- VAE canonical test MSE
- Transformer best validation MSE

This is not a fair direct comparison.

Required fix:
create a canonical evaluation script for Transformer V2 that runs on the same:

`data/splits/test.csv`

Expected test counts:
- total: 1,892
- authentic: 1,123
- tampered: 769

Calculate:
- MSE
- PSNR
- SSIM
- per-image results
- overall/authentic/tampered summaries
- evaluation time
- optional reconstruction-error ROC-AUC, clearly marked exploratory

Suggested new file:
`src/evaluate_transformer_v2.py`

Suggested outputs:
- `results/transformer_v2_test_metrics.json`
- `results/transformer_v2_test_per_image_metrics.csv`
- selected reconstruction grid/plots

Only after this evaluation should the GUI/PPT directly compare AE/VAE/Vision Transformer reconstruction.

---

# 7. New prompt/text Transformer evidence-intelligence module

This is the largest new Review-2 feature.

## 7.1 Why it is needed
Faculty explicitly expects a Transformer demonstration involving:
- prompt;
- tokenization;
- LLM/SLM behavior;
- text/PDF evidence;
- generated response.

This should be added without deleting the existing Vision Transformer.

## 7.2 Recommended architecture
Create a separate module:

`Text/PDF Evidence`
-> `safe document text extraction`
-> `cleaning/chunking`
-> `Tokenizer`
-> `pretrained Transformer language model`
-> `User prompt`
-> `Generated Evidence Intelligence`

## 7.3 Recommended implementation direction
Prefer an open-source Hugging Face model with no mandatory API key.

A practical default for CPU/Streamlit deployment is a FLAN-T5 family model.

Suggested design:
- default lightweight model configurable by environment variable;
- e.g. `google/flan-t5-small` for portability;
- allow model ID configuration if a stronger local machine/GPU is available.

Important:
- clearly state it is pretrained;
- do not claim the team trained an LLM from scratch;
- retain the trained Vision Transformer as the course implementation.

## 7.4 Suggested new modules
- `src/document_parser.py`
- `src/evidence_transformer.py`
- `src/evidence_inference.py`

Possible dependencies:
- `transformers`
- `sentencepiece`
- `pypdf`

Add dependencies carefully to `requirements.txt`.

## 7.5 Minimum supported evidence inputs
For first Review-2 version:
- pasted text;
- `.txt`;
- digitally extractable `.pdf`.

If PDF text extraction returns empty:
- report that it may be a scanned/image-only PDF;
- do not pretend OCR succeeded;
- OCR can be added later if needed.

## 7.6 Prompt UI
Provide:
- evidence upload or pasted text;
- extracted text preview;
- text length/token estimate;
- prompt textbox;
- a professional default prompt template;
- Generate button;
- generated output;
- runtime/inference metadata;
- responsible-use warning.

Example default prompt intent:
- summarize important facts;
- identify important entities;
- identify possible inconsistencies or unusual statements;
- explain why each observation may matter;
- state uncertainty/limitations;
- avoid declaring guilt, authenticity, or a legal conclusion.

## 7.7 Long document handling
Do not silently truncate large documents without telling the user.

Implement basic chunking:
- tokenize/chunk within model limits;
- process chunks;
- combine/summarize results;
- show how much text was processed.

Keep the first implementation simple and reliable.

## 7.8 Responsible output
Generated intelligence must be framed as AI-assisted analysis, not a forensic verdict.

Include:
- "Research/educational use"
- "Verify against original evidence"
- "Generated text may contain errors"
- "Do not use as sole legal/forensic basis"

---

# 8. Evaluation strategy

## 8.1 Image reconstruction comparison
After canonical Transformer evaluation, create one authoritative comparison artifact using the same 1,892-image test split.

Columns should include:
- Model
- Primary role
- Test images
- MSE
- PSNR
- SSIM
- Parameters
- Best/checkpoint epoch
- evaluation time if available

Use sensible rounding, e.g. four decimals where appropriate.

Do not use a validation metric in a table labelled test comparison.

## 8.2 Anomaly metrics
If ROC-AUC is shown:
- label it exploratory;
- do not call it accuracy of forgery detection;
- note that values near 0.5 are weak.

## 8.3 Text Transformer
Do not compare generated text to AE/VAE using MSE/PSNR/SSIM.

For the Review-2 text module, useful demo metadata may include:
- input characters/tokens;
- output tokens;
- inference time;
- model name;
- generation settings.

If no reference-answer dataset exists, do not invent ROUGE/BLEU/accuracy values.

---

# 9. Professional GUI target

Recommended navigation:

1. Project Overview
2. Autoencoder
3. Variational Autoencoder
4. Vision Transformer
5. Evidence Intelligence Transformer
6. Model Evaluation
7. Privacy & Ethical AI
8. Project Information

## 9.1 Project Overview
Show:
- project title;
- current three required architectures;
- CASIA dataset counts;
- current workflow;
- clear responsible-use statement.

## 9.2 AE page
Show:
- architecture summary;
- Original Image;
- Reconstructed Image;
- MSE;
- PSNR;
- SSIM;
- checkpoint epoch;
- test metrics;
- no classification verdict.

## 9.3 VAE page
Show:
- architecture;
- mu/logvar concept;
- deterministic reconstruction;
- stochastic reconstructions;
- temperature;
- relevant metrics;
- KL explanation;
- warning that stochastic variations are model samples, not discovered truth.

## 9.4 Vision Transformer page
Keep:
- image upload;
- reconstruction;
- canonical metrics once available;
- attention map/overlay;
- warning against over-interpreting attention.

## 9.5 Evidence Intelligence Transformer page
Show:
- PDF/text evidence;
- extracted text;
- prompt;
- tokenization/model information;
- generated evidence summary/observations;
- responsible-use note.

## 9.6 Model Evaluation page
Separate:
- image reconstruction comparison;
- architecture/role comparison;
- text Transformer runtime/demo information.

## 9.7 Privacy & Ethical AI page
Implement actual controls:
- user consent checkbox before evidence processing;
- file extension validation;
- file size limit;
- temporary/in-memory processing where possible;
- no automatic permanent evidence storage;
- clear cleanup statement;
- privacy notice;
- responsible-use disclaimer;
- no legal/forensic verdict;
- explain applicable GDPR-oriented design principles such as:
  - data minimization;
  - purpose limitation;
  - transparency;
  - storage limitation;
  - integrity/confidentiality.

Do **not** claim formal GDPR compliance unless established.

---

# 10. Repository/documentation issues already identified

## 10.1 README contradictions
Current README contains inconsistent/outdated statements, including:
- one section calling VAE V5 authentic-only;
- another calling the current checkpoint mixed-data;
- an old VAE metric row (`0.00106170`) instead of current final `0.00070007`;
- a statement that Transformer is not implemented even though Transformer V2 is implemented.

Do not finalize README until the Review-2 implementation is stable.

Then clean these contradictions using actual verified facts only.

## 10.2 Generated image artifacts
`.gitignore` ignores:
`outputs/`

README references some output images that are therefore absent from the GitHub ZIP.

Create a small committed folder such as:
`docs/results/`

Store only selected presentation-ready artifacts there, for example:
- AE reconstruction examples;
- AE training curve;
- VAE deterministic + stochastic examples;
- Vision Transformer reconstruction/attention example;
- prompt/text Transformer output screenshot;
- final model comparison chart/table.

Do not commit huge generated folders.

## 10.3 VAE LFS
Handle the VAE LFS pointer issue before claiming the GitHub ZIP is directly runnable.

---

# 11. Recommended implementation phases

## Phase 0 — Repository audit and safety
Before editing:
- inspect `git status`;
- inspect current file tree;
- verify available checkpoints;
- identify Git LFS pointer files;
- run import/syntax smoke tests;
- create a concise baseline report.

Do not make destructive changes.

## Phase 1 — Finalize Autoencoder
Tasks:
- fix `load_ae()` to use `best_autoencoder_rtx80_portable.pth`;
- update any RTX80 evaluation defaults as needed;
- rename UI headings to Original Image / Reconstructed Image;
- verify metrics shown match RTX80 files;
- run real inference;
- optionally add batch/sample-demo support if useful;
- ensure AE page is Review-2 ready.

Phase acceptance:
- checkpoint correct;
- inference works;
- labels correct;
- metrics correct;
- no false detector claim.

## Phase 2 — VAE probabilistic demonstration
Tasks:
- add image-conditioned stochastic inference;
- add multiple variations to GUI;
- add seed/temperature controls carefully;
- verify deterministic metrics remain unchanged;
- evaluate whether stochastic variation is visibly meaningful;
- if not, quantify the issue and propose/prepare retraining;
- never fake visible variation.

Phase acceptance:
- mathematically correct reparameterized samples;
- output visibly/quantitatively demonstrates probabilistic behavior or documented model limitation;
- deterministic VAE remains stable.

## Phase 3 — Prompt/Text Transformer
Tasks:
- implement PDF/text parser;
- implement pretrained Transformer model wrapper;
- tokenization;
- prompt UI;
- chunking/length handling;
- generated evidence-intelligence output;
- responsible-use notices;
- CPU-safe smoke test;
- clear separation from Vision Transformer.

Phase acceptance:
- text and at least one extractable PDF can be processed;
- prompt produces real model-generated output;
- model/tokenizer names visible;
- no API key required for basic path;
- limitations explicit.

## Phase 4 — Canonical Vision Transformer evaluation
Tasks:
- implement `evaluate_transformer_v2.py`;
- use canonical 1,892 test split;
- calculate MSE/PSNR/SSIM;
- save per-image CSV and summary JSON;
- optional exploratory ROC-AUC;
- update comparison data only after real results exist.

Phase acceptance:
- exact test counts match;
- metrics reproducible;
- no validation/test mixing.

## Phase 5 — Unified evaluation/comparison
Tasks:
- build one authoritative comparison table;
- load values from result files, not manually duplicated constants where practical;
- sensible rounding;
- clearly separate image reconstruction metrics from language-model information.

## Phase 6 — Professional GUI
Tasks:
- reorganize navigation;
- improve consistent labels/cards;
- add model result context;
- no overly long theory;
- clear error handling;
- current model/checkpoint information;
- 20–25 sample readiness via repeatable upload/test process or optional sample gallery if legitimate sample assets are available.

## Phase 7 — Privacy/Ethical deployment
Tasks:
- consent;
- file validation;
- size limits;
- temporary processing;
- privacy notice;
- responsible-use;
- GDPR-oriented principles;
- deployment readiness.

## Phase 8 — Repository cleanup
Tasks:
- selected `docs/results/`;
- fix README;
- requirements;
- checkpoint download/LFS strategy;
- remove stale contradictions;
- keep GAN preserved;
- Diffusion still future.

## Phase 9 — PPT support only after code/results stabilize
Prepare code-generated facts/tables/screenshots for PPT.
Do not invent literature references.
The user will handle/finalize the PPT separately after implementation.

---

# 12. Current verified values that must not be accidentally replaced

## AE RTX-80 Final
- input: 3x128x128
- latent: 32x8x8
- compression: 24x
- parameters: 265,571
- best epoch: 76
- val MSE: 0.00305005
- test images: 1,892
- test MSE: 0.00303199
- test PSNR: 26.0159 dB
- test SSIM: 0.792911

## VAE V5 Final
- latent: 256
- parameters: 17,599,971
- checkpoint epoch: 79
- test MSE: 0.00070007
- test PSNR: 32.9535 dB
- test SSIM: 0.961139
- mean KL: 0.01292955
- MSE ROC-AUC: 0.5154
- SSIM-error ROC-AUC: 0.5706

## Vision Transformer V2
- image size: 128
- patch: 16x16
- tokens: 64
- embedding: 256
- heads: 8
- encoder layers: 4
- decoder layers: 2
- feed-forward: 512
- epoch: 50
- train MSE: 0.00256850
- validation MSE: 0.00206004
- saved ROC-AUC: 0.5455
- canonical 1,892-image test MSE/PSNR/SSIM: NOT YET VERIFIED

Never fill the final Transformer test row with validation MSE.

---

# 13. What must not be claimed

Do not claim:
- AE detects tampering;
- VAE detects forgery;
- reconstruction error is tampering probability;
- attention map is definitive manipulation localization;
- VAE prior generation is strong when it is known to be weak;
- Transformer validation MSE is a canonical test metric;
- prompt Transformer was trained from scratch if a pretrained model is used;
- Diffusion is implemented;
- GAN is part of the current three-model GUI unless re-added intentionally later;
- the project is formally GDPR compliant;
- the system makes legal/forensic decisions.

---

# 14. Definition of Review-2-ready

The repository is Review-2-ready only when:

- AE uses the correct improved checkpoint and displays clean reconstruction correctly.
- VAE visibly demonstrates mathematically correct probabilistic sampling for an uploaded image.
- Existing Vision Transformer remains functional.
- Prompt/PDF Transformer evidence-intelligence module works.
- Vision Transformer has canonical test evaluation on the same 1,892 test images.
- Image model comparison uses compatible test metrics.
- GUI is professional and clearly separates model roles.
- Privacy/ethical controls are implemented.
- checkpoint/LFS failures are handled clearly.
- README reflects actual final state.
- selected result screenshots/artifacts are available for PPT/demo.
- no fabricated metric or exaggerated forensic claim remains.
