"""Professional Streamlit interface for the Digital Evidence GenAI framework."""

from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ae_inference import AutoencoderInference
from document_parser import DocumentParseError, MAX_EVIDENCE_BYTES, parse_evidence_file
from evidence_transformer import DEFAULT_PROMPT, EvidenceModelError, EvidenceTransformer
from transformer_inference import TransformerInference
from vae_inference import VAEInference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="Digital Evidence GenAI", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {--navy:#071b2d;--cyan:#11a8c7;--ink:#122333;--muted:#5d7182;--line:#dbe7ef;--soft:#f4f8fb;}
.block-container{max-width:1380px;padding-top:1.35rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071b2d 0%,#0b3658 100%)}
[data-testid="stSidebar"] *{color:#edf7fb}[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.18)}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:9px;padding:.28rem .45rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(255,255,255,.09)}
.hero{position:relative;overflow:hidden;padding:2rem 2.2rem;border-radius:22px;background:linear-gradient(125deg,#071b2d 0%,#0b4f78 58%,#0b8298 100%);box-shadow:0 18px 45px rgba(7,27,45,.18);color:white;margin-bottom:1.4rem}
.hero:after{content:"";position:absolute;width:280px;height:280px;right:-70px;top:-120px;border-radius:50%;border:45px solid rgba(255,255,255,.07)}
.eyebrow,.section-kicker,.small-label{font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.eyebrow{color:#8fe8f2}.hero h1{font-size:2.25rem;line-height:1.14;margin:.45rem 0 .7rem;max-width:920px}.hero p{font-size:1rem;line-height:1.65;max-width:880px;color:#d9edf5;margin:0}
.section-kicker{color:#0b7890;margin-bottom:.2rem}.section-title{font-size:1.65rem;font-weight:750;color:var(--ink);margin:0 0 .35rem}.section-copy{color:var(--muted);max-width:900px;line-height:1.6;margin-bottom:1.1rem}
.model-card{border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.2rem;background:white;box-shadow:0 7px 22px rgba(16,52,77,.07);min-height:182px}.model-card .tag{display:inline-block;padding:.22rem .58rem;border-radius:999px;background:#e5f7f8;color:#08758b;font-size:.72rem;font-weight:700;text-transform:uppercase}.model-card h3{color:var(--ink);margin:.7rem 0 .35rem;font-size:1.25rem}.model-card p{color:var(--muted);line-height:1.5;margin:0}
.workflow{border:1px solid var(--line);border-radius:16px;background:var(--soft);padding:1rem 1.2rem;color:#24445b;font-weight:600;text-align:center}.guardrail{border-left:5px solid #e59a2f;background:#fff8eb;border-radius:10px;padding:.9rem 1rem;color:#68420c;line-height:1.55;margin:.8rem 0 1rem}.success-note{border-left:5px solid #18a37a;background:#ecfbf5;border-radius:10px;padding:.8rem 1rem;color:#155d49}.upload-guide{border:1px dashed #9ab6c8;border-radius:15px;padding:1.2rem;background:#f7fafc;color:#526c7d;text-align:center;margin:.5rem 0}.small-label{color:#668091}
div[data-testid="stMetric"]{border:1px solid var(--line);border-radius:14px;padding:.75rem 1rem;background:linear-gradient(180deg,#fff,#f7fafc);box-shadow:0 4px 13px rgba(16,52,77,.05)}
.stButton>button{border-radius:10px;min-height:2.7rem;font-weight:700}@media(max-width:800px){.hero{padding:1.4rem}.hero h1{font-size:1.7rem}}
</style>""", unsafe_allow_html=True)


def configured_checkpoint(environment_name: str, default_path: str) -> Path:
    configured = Path(os.getenv(environment_name, default_path))
    return configured if configured.is_absolute() else ROOT / configured


@st.cache_resource(show_spinner="Loading Autoencoder…")
def load_ae():
    return AutoencoderInference(
        configured_checkpoint("AE_CHECKPOINT_PATH", "checkpoints/best_quality_autoencoder_v1.pth"),
        DEVICE,
    )


@st.cache_resource(show_spinner="Loading VAE V5 Final…")
def load_vae():
    return VAEInference(
        configured_checkpoint("VAE_CHECKPOINT_PATH", "checkpoints/VAE_V5_FINAL.pth"), DEVICE, 256
    )


@st.cache_resource(show_spinner="Loading Transformer V2…")
def load_transformer():
    return TransformerInference(
        configured_checkpoint("VISION_TRANSFORMER_CHECKPOINT_PATH", "checkpoints/transformer_v2_final.pth"),
        DEVICE,
    )


@st.cache_resource(show_spinner="Loading pretrained evidence Transformer…")
def load_evidence_transformer():
    return EvidenceTransformer(device=DEVICE)


@st.cache_data
def load_image_comparison() -> pd.DataFrame:
    result_path = ROOT / "results" / "image_model_comparison.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload["models"]
    return pd.DataFrame(
        {
            "Model": [row["model"] for row in rows],
            "Primary role": [row["primary_role"] for row in rows],
            "Test images": [f"{int(row['test_images']):,}" for row in rows],
            "MSE ↓": [f"{float(row['mse']):.6f}" for row in rows],
            "PSNR ↑": [f"{float(row['psnr_db']):.4f} dB" for row in rows],
            "SSIM ↑": [f"{float(row['ssim']):.6f}" for row in rows],
            "Parameters": [f"{int(row['parameters']):,}" for row in rows],
            "Epoch": [int(row["checkpoint_epoch"]) for row in rows],
        }
    )


@st.cache_data
def load_result(filename: str) -> dict:
    return json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))


@st.cache_data
def load_demo_manifest() -> list[dict[str, str]]:
    manifest_path = ROOT / "data" / "splits" / "test.csv"
    if not manifest_path.is_file():
        return []
    frame = pd.read_csv(manifest_path)
    selected = pd.concat(
        [frame[frame["class_name"] == class_name].head(12) for class_name in ("authentic", "tampered")],
        ignore_index=True,
    )
    records: list[dict[str, str]] = []
    for _, row in selected.iterrows():
        image_path = Path(str(row["image_path"]))
        resolved = image_path if image_path.is_absolute() else ROOT / image_path
        if resolved.is_file():
            records.append(
                {
                    "label": f"{str(row['class_name']).title()} · {resolved.name}",
                    "path": str(resolved),
                }
            )
    return records


def safe_load(loader, label):
    try:
        return loader(), None
    except FileNotFoundError as exc:
        return None, f"{label} checkpoint is missing: {exc}"
    except Exception as exc:
        return None, f"{label} could not be loaded: {exc}"


def uploaded_image(uploaded):
    if uploaded is None:
        return None
    if Path(uploaded.name).suffix.lower() not in SUPPORTED_SUFFIXES:
        st.error("Unsupported format. Upload JPG, JPEG, PNG, BMP, TIF, or TIFF.")
        return None
    if uploaded.size > MAX_IMAGE_BYTES:
        st.error(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB in-memory limit.")
        return None
    try:
        image = Image.open(uploaded)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"This image could not be read safely: {exc}")
        return None


def heading(kicker, title, copy):
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div><div class="section-copy">{copy}</div>', unsafe_allow_html=True)


def metrics_cards(metrics):
    columns = st.columns(3)
    columns[0].metric("MSE ↓", f"{metrics['mse']:.6f}", help="Lower means less pixel error.")
    columns[1].metric("PSNR ↑", f"{metrics['psnr']:.3f} dB", help="Higher generally means better reconstruction.")
    columns[2].metric("SSIM ↑", f"{metrics['ssim']:.4f}", help="Closer to 1 means stronger structural similarity.")


def compare_images(original, reconstructed, caption):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="small-label">Original Image</div>', unsafe_allow_html=True)
        st.image(original, caption="Original Image", width="stretch")
    with right:
        st.markdown('<div class="small-label">Reconstructed Image</div>', unsafe_allow_html=True)
        st.image(reconstructed, caption=caption, width="stretch", clamp=True)


def model_specification(rows: list[tuple[str, str]]) -> None:
    """Render the faculty-requested model description in a compact, consistent form."""
    with st.expander("Architecture, objective, and role", expanded=False):
        st.dataframe(
            pd.DataFrame(rows, columns=["Aspect", "Current implementation"]),
            hide_index=True,
            width="stretch",
        )


def image_input(key):
    demo_records = load_demo_manifest()
    choices = ["Upload image"] + (["Canonical test demo"] if demo_records else [])
    source = st.radio("Image source", choices, horizontal=True, key=f"{key}_source")
    if source == "Canonical test demo":
        with st.expander(f"Browse available held-out gallery ({len(demo_records)} samples)"):
            gallery_columns = st.columns(6)
            for index, record in enumerate(demo_records):
                with gallery_columns[index % 6]:
                    st.image(record["path"], width="stretch")
                    st.caption(record["label"])
        labels = [record["label"] for record in demo_records]
        selected_label = st.selectbox(
            f"Choose one of {len(demo_records)} reproducible held-out examples",
            labels,
            key=f"{key}_demo",
        )
        selected = demo_records[labels.index(selected_label)]
        try:
            with Image.open(selected["path"]) as opened:
                image = opened.convert("RGB").copy()
            st.caption(f"{selected_label} · canonical test manifest · {image.width}×{image.height} · RGB")
            return image
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"The selected demo image could not be read: {exc}")
            return None

    if not demo_records:
        st.caption(
            "The local CASIA files are unavailable, so the held-out gallery is disabled. "
            "Manual upload remains available; the full dataset is intentionally not committed."
        )

    upload = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key=key,
        help=f"Supported images up to {MAX_IMAGE_BYTES // (1024 * 1024)} MB; processed in memory.",
    )
    image = uploaded_image(upload)
    if image is None:
        st.markdown(
            '<div class="upload-guide">Upload a supported image to begin reconstruction and metric analysis.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"{upload.name} · original size {image.width}×{image.height} · RGB")
    return image


def verified_test_cards(filename: str, count_key: str) -> None:
    try:
        result = load_result(filename)
        overall = result["overall"]
        cards = st.columns(4)
        cards[0].metric("Canonical test images", f"{int(result[count_key]):,}")
        cards[1].metric("Test MSE ↓", f"{float(overall['mse_mean']):.6f}")
        cards[2].metric("Test PSNR ↑", f"{float(overall['psnr_mean']):.4f} dB")
        cards[3].metric("Test SSIM ↑", f"{float(overall['ssim_mean']):.6f}")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Verified test summary is unavailable: {exc}")


def processing_consent(key: str) -> bool:
    return st.checkbox(
        "I am authorized to process this evidence and consent to in-session analysis.",
        key=f"{key}_processing_consent",
        help="Only submit evidence you are permitted to use. This prototype does not establish legal authority.",
    )


def clear_session_evidence() -> int:
    prefixes = ("ae_", "vae_", "transformer_", "evidence_")
    keys = [key for key in st.session_state if key.startswith(prefixes)]
    for key in keys:
        del st.session_state[key]
    return len(keys)


with st.sidebar:
    st.markdown("## 🔬 Digital Evidence AI")
    st.caption("Semester 7 · Generative AI")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Project Overview",
            "Autoencoder",
            "Variational Autoencoder",
            "Vision Transformer",
            "Evidence Intelligence Transformer",
            "Model Evaluation",
            "Privacy & Ethical AI",
            "Project Information",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Runtime device**")
    st.caption(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    st.caption("CUDA enabled" if torch.cuda.is_available() else "CUDA unavailable · CPU inference")
    st.divider()
    st.caption("CASIA v2.0 · 12,614 images")
    st.caption("Academic research prototype")

st.markdown("""<div class="hero"><div class="eyebrow">Multi-model generative AI framework</div><h1>Digital Evidence Analysis and Intelligence Generation</h1><p>Deterministic, probabilistic, and attention-based image reconstruction plus pretrained Transformer assistance for text and PDF evidence.</p></div>""", unsafe_allow_html=True)


if page == "Project Overview":
    heading("Project foundation", "A multi-model evidence research framework", "Three trained image architectures support reconstruction research, while a separate pretrained language Transformer assists with text evidence. The application does not issue forensic verdicts.")
    counts = st.columns(4)
    for column, label, value in zip(counts, ["CASIA images", "Authentic", "Tampered", "Held-out test"], ["12,614", "7,491", "5,123", "1,892"]):
        column.metric(label, value)
    st.markdown("### Current model modules")
    cards = st.columns(4, gap="large")
    cards[0].markdown('<div class="model-card"><span class="tag">Deterministic</span><h3>Quality Autoencoder</h3><p>Residual high-fidelity reconstruction through a genuine 6Ã— latent bottleneck.</p></div>', unsafe_allow_html=True)
    cards[1].markdown('<div class="model-card"><span class="tag">Probabilistic</span><h3>VAE V5 Final</h3><p>256-dimensional probabilistic representation with high-quality skip-connected reconstruction.</p></div>', unsafe_allow_html=True)
    cards[2].markdown('<div class="model-card"><span class="tag">Attention</span><h3>Transformer V2</h3><p>Patch-token reconstruction with interpretable learned attention relationships.</p></div>', unsafe_allow_html=True)
    cards[3].markdown('<div class="model-card"><span class="tag">Pretrained SLM</span><h3>Evidence Intelligence</h3><p>Token-aware assistance for pasted text and digitally extractable PDF evidence.</p></div>', unsafe_allow_html=True)
    st.markdown("### Analysis workflow")
    st.markdown('<div class="workflow">Upload → RGB preprocessing → reconstruction → MSE · PSNR · SSIM → exploratory interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="guardrail"><b>Interpretation boundary:</b> reconstruction error, thresholds, difference maps, and attention are not proof of manipulation and must not be treated as legal or forensic conclusions.</div>', unsafe_allow_html=True)
    with st.expander("Dataset safeguards and reproducibility"):
        st.write("Fixed stratified split: 70% train, 15% validation, 15% test; seed 42.")
        st.write("Ground-truth mask PNG files are excluded from normal model inputs.")
        st.write("All inputs are RGB 128×128 tensors normalized to [0,1].")


elif page == "Autoencoder":
    heading("Model 01 · deterministic reconstruction", "Quality Autoencoder V1", "Examine high-fidelity deterministic reconstruction from a residual Autoencoder with a genuine 6× latent bottleneck. Metrics compare input and reconstruction; the model does not predict a label.")
    info = st.columns(4)
    for column, label, value in zip(info, ["Input", "Latent", "Compression", "Parameters"], ["3×128×128", "32×16×16", "6×", "2,371,715"]):
        column.metric(label, value)
    st.markdown('<div class="workflow">RGB image → residual encoder → 6× latent bottleneck → resize-convolution decoder → reconstruction</div>', unsafe_allow_html=True)
    st.markdown("#### Verified canonical test result")
    verified_test_cards("quality_ae_v1_test_metrics.json", "number_of_test_images")
    model_specification(
        [
            ("Architecture", "Residual convolutional Autoencoder without encoder-decoder skip bypasses"),
            ("Input", "128×128 RGB image normalized to [0, 1]"),
            ("Processing", "Residual encoder → 32×16×16 bottleneck → resize-convolution decoder"),
            ("Output", "Deterministic 128×128 RGB reconstruction"),
            ("Loss/objective", "0.65 L1 + 0.25 (1 − SSIM) + 0.10 edge-preservation loss"),
            ("Evaluation", "MSE, PSNR, and SSIM on the canonical 1,892-image test split"),
            ("Role", "High-fidelity deterministic compression/reconstruction; not a forgery classifier"),
        ]
    )
    model, error = safe_load(load_ae, "Autoencoder")
    if error:
        st.error(error)
    else:
        image = image_input("ae_upload") if processing_consent("ae") else None
        if image is None and not st.session_state.get("ae_processing_consent", False):
            st.info("Confirm authorization and consent to enable evidence input.")
        if image is not None:
            try:
                with st.spinner("Reconstructing image…"):
                    original, reconstructed, result = model.reconstruct(image)
                compare_images(original, reconstructed, "Autoencoder reconstruction")
                metrics_cards(result)
                st.markdown('<div class="success-note">Reconstruction fidelity only—no authenticity or tampering class was produced.</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Autoencoder inference failed: {exc}")
    with st.expander("Architecture and interpretation"):
        st.write("The residual encoder compresses 49,152 RGB values into an 8,192-value latent tensor. The decoder uses bilinear resize-convolution blocks and Sigmoid output; no encoder-to-decoder skip connection bypasses the bottleneck.")
        st.write("Lower MSE is better; higher PSNR is better; SSIM closer to 1 indicates stronger structural similarity.")
        st.write("Current checkpoint: epoch 79 selected by validation SSIM 0.942674 · validation MSE 0.00095790.")
        st.write("Complete held-out test: MSE 0.00094603 · PSNR 31.1951 dB · SSIM 0.938654.")
        st.caption("The preserved RTX-80 model used a stronger 24× compression ratio; this quality-focused model trades compression for substantially better reconstruction fidelity.")


elif page == "Variational Autoencoder":
    heading("Model 02 · probabilistic reconstruction", "Variational Autoencoder V5 Final", "Compare the deterministic posterior mean with genuine image-conditioned samples from the learned 256-dimensional latent distribution.")
    info = st.columns(4)
    for column, label, value in zip(info, ["Input", "Latent", "Parameters", "Checkpoint epoch"], ["3×128×128", "256", "17,599,971", "79"]):
        column.metric(label, value)
    st.markdown('<div class="workflow">Encoder → μ and log variance → latent z → skip-connected decoder → reconstruction</div>', unsafe_allow_html=True)
    st.markdown("#### Verified canonical test result")
    verified_test_cards("vae_v5_final_test_metrics.json", "test_images")
    model_specification(
        [
            ("Architecture", "Residual VAE V5 with GroupNorm, SiLU, 256-D latent space, and decoder skips"),
            ("Input", "128×128 RGB image normalized to [0, 1]"),
            ("Processing", "Encoder → μ/log variance → reparameterized z → skip-connected decoder"),
            ("Output", "Deterministic posterior-mean reconstruction and three posterior samples"),
            ("Loss/objective", "0.5 MSE + 0.5 L1 reconstruction loss + beta-weighted KL divergence"),
            ("Evaluation", "MSE, PSNR, SSIM, and mean KL on the canonical held-out test split"),
            ("Role", "Probabilistic latent representation and reconstruction; not a tampering verdict"),
        ]
    )
    model, error = safe_load(load_vae, "VAE V5 Final")
    if error:
        st.error(error)
    else:
        reconstruction_tab, generation_tab = st.tabs(["Probabilistic reconstructions", "Prior-only limitation"])
        with reconstruction_tab:
            controls = st.columns(2)
            temperature = controls[0].slider(
                "Sampling temperature", 0.25, 3.0, 2.0, 0.25,
                help="Scales posterior uncertainty; higher values produce larger latent changes.",
            )
            sample_seed = controls[1].number_input(
                "Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1
            )
            image = image_input("vae_upload") if processing_consent("vae") else None
            if image is None and not st.session_state.get("vae_processing_consent", False):
                st.info("Confirm authorization and consent to enable evidence input.")
            if image is not None:
                try:
                    with st.spinner("Sampling the image-conditioned VAE posterior…"):
                        original, reconstructed, variations, result = model.reconstruct_variations(
                            image,
                            n_samples=3,
                            temperature=float(temperature),
                            seed=int(sample_seed),
                        )
                    compare_images(original, reconstructed, "Deterministic Reconstruction (z = μ)")
                    metrics_cards(result)
                    st.markdown("#### Probabilistic latent variations")
                    variation_columns = st.columns(len(variations))
                    for index, (column, variation) in enumerate(zip(variation_columns, variations), 1):
                        with column:
                            st.markdown(
                                f'<div class="small-label">Stochastic Variation {index}</div>',
                                unsafe_allow_html=True,
                            )
                            st.image(
                                variation["image"],
                                caption=f"z = μ + {temperature:.2f}σε",
                                width="stretch",
                                clamp=True,
                            )
                            st.caption(
                                "Mean |pixel delta| from deterministic: "
                                f"{variation['mean_abs_delta_from_deterministic']:.6f}"
                            )
                    st.markdown("#### Difference maps: stochastic variation − deterministic reconstruction")
                    difference_columns = st.columns(3)
                    for index, (column, variation) in enumerate(zip(difference_columns, variations), 1):
                        with column:
                            figure, axis = plt.subplots(figsize=(4, 3.3))
                            heatmap = axis.imshow(variation["difference_map"], cmap="magma")
                            axis.set_title(f"Variation {index} · absolute RGB difference")
                            axis.axis("off")
                            figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
                            figure.tight_layout()
                            st.pyplot(figure)
                            plt.close(figure)
                            st.caption(
                                f"Maximum mean-channel difference: {variation['difference_map_max']:.6f}"
                            )
                    st.caption(
                        "Each heatmap uses its own measured color scale so subtle posterior effects remain "
                        "visible. The reconstructed images themselves are not enhanced or modified."
                    )
                    st.caption(
                        f"Reproducible posterior samples · seed {int(sample_seed)} · "
                        f"temperature {temperature:.2f}. Small visual differences may reflect the "
                        "checkpoint's strong skip connections."
                    )
                    st.markdown('<div class="guardrail"><b>Exploratory indicator:</b> MSE is reconstruction error—not a tampering probability or standalone decision.</div>', unsafe_allow_html=True)
                    st.info(
                        "These are probabilistic latent reconstructions conditioned on the uploaded image. "
                        "They are model samples—not newly discovered forensic features or evidence of manipulation."
                    )
                except Exception as exc:
                    st.error(f"VAE posterior sampling failed: {exc}")
        with generation_tab:
            st.info("VAE V5 was optimized with encoder skip features. Random prior samples lack those features and may appear dark or visually degenerate.")
            if st.button("Generate experimental latent sample", type="primary"):
                try:
                    with st.spinner("Sampling z ~ N(0,I)…"):
                        generated = model.generate()
                    st.image(generated, caption="Synthetic VAE research output", width=430, clamp=True)
                    st.warning("Synthetic research output—not genuine forensic evidence. Weak prior generation is a documented limitation.")
                except Exception as exc:
                    st.error(f"VAE generation failed: {exc}")
    with st.expander("Current evaluation context"):
        st.write("Canonical test: MSE 0.00070007 · PSNR 32.9535 dB · SSIM 0.961139.")
        st.write("MSE ROC-AUC 0.5154 indicates almost no useful tampering ranking by reconstruction error alone.")


elif page == "Vision Transformer":
    heading("Model 03 · patch attention", "Transformer Forensic Autoencoder V2", "The image becomes 64 patch tokens. Self-attention models patch relationships while all encoded spatial tokens are retained for reconstruction.")
    info = st.columns(4)
    for column, label, value in zip(info, ["Input", "Patch size", "Patch tokens", "Embedding"], ["3×128×128", "16×16", "64", "256"]):
        column.metric(label, value)
    layers = st.columns(3)
    for column, label, value in zip(layers, ["Attention heads", "Encoder layers", "Decoder layers"], ["8", "4", "2"]):
        column.metric(label, value)
    st.markdown('<div class="workflow">Image → 8×8 patch grid → attention encoder → token decoder → reconstruction + attention view</div>', unsafe_allow_html=True)
    st.markdown("#### Verified canonical test result")
    verified_test_cards("transformer_v2_test_metrics.json", "test_images")
    model_specification(
        [
            ("Architecture", "Vision Transformer autoencoder: 64 patch tokens, 256-D embeddings, 8 heads"),
            ("Input", "128×128 RGB image split into an 8×8 grid of 16×16 patches"),
            ("Processing", "Patch embedding + position embedding → 4 encoder layers → 2 decoder layers"),
            ("Output", "RGB reconstruction plus exploratory 8×8 attention visualization"),
            ("Loss/objective", "Mean squared reconstruction error"),
            ("Evaluation", "MSE, PSNR, SSIM, and exploratory ROC-AUC on the canonical test split"),
            ("Role", "Patch-relationship reconstruction and attention exploration; not manipulation proof"),
        ]
    )
    model, error = safe_load(load_transformer, "Transformer V2")
    if error:
        st.error(error)
    else:
        image = image_input("transformer_upload") if processing_consent("transformer") else None
        if image is None and not st.session_state.get("transformer_processing_consent", False):
            st.info("Confirm authorization and consent to enable evidence input.")
        if image is not None:
            try:
                with st.spinner("Running patch-attention reconstruction…"):
                    original, reconstructed, result, latent, attention = model.reconstruct(image)
                compare_images(original, reconstructed, "Transformer V2 reconstruction")
                metrics_cards(result)
                st.markdown("#### Exploratory reconstruction indicator")
                indicator = st.columns(3)
                indicator[0].metric("Reconstruction error", f"{result['reconstruction_error']:.6f}")
                indicator[1].metric("Reference threshold", f"{result['anomaly_threshold']:.6f}")
                status = "Above reference range" if result["reconstruction_error"] > result["anomaly_threshold"] else "Within reference range"
                indicator[2].metric("Indicator", status)
                st.markdown("#### Patch-attention visualization")
                figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
                axes[0].imshow(attention, cmap="viridis"); axes[0].set_title("8×8 patch attention")
                overlay = Image.fromarray((attention * 255).astype("uint8")).resize((128, 128), Image.Resampling.BILINEAR)
                axes[1].imshow(original.resize((128, 128))); axes[1].imshow(overlay, cmap="jet", alpha=.45); axes[1].set_title("Attention overlay")
                for axis in axes:
                    axis.axis("off")
                figure.tight_layout(); st.pyplot(figure); plt.close(figure)
                st.markdown('<div class="guardrail"><b>Do not over-interpret:</b> this is not a tampering probability. Attention shows learned patch relationships and does not definitively localize manipulation.</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Transformer inference failed: {exc}")
    with st.expander("Checkpoint and evaluation context"):
        st.write("Epoch 50 · final training MSE 0.00256850 · best validation MSE 0.00206004.")
        st.write("Saved reconstruction-error ROC-AUC 0.5455 indicates limited separation.")


elif page == "Evidence Intelligence Transformer":
    heading(
        "Pretrained language model integration",
        "Text/PDF Evidence Intelligence Transformer",
        "Extract text, inspect tokenization, and use a no-key pretrained FLAN-T5 model to generate review-oriented observations. This module is separate from the trained Vision Transformer.",
    )
    st.markdown(
        '<div class="workflow">TXT or digital PDF → in-memory extraction → token-aware chunks → tokenizer → pretrained Transformer → evidence-oriented output</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "The language model is pretrained and integrated for this project; it was not trained from scratch by the team. "
        "The first model load may download open-source weights."
    )
    model_specification(
        [
            ("Architecture", "Pretrained google/flan-t5-small encoder-decoder Transformer integration"),
            ("Input", "Pasted text, UTF-8 TXT, or digitally extractable PDF up to 10 MB"),
            ("Processing", "In-memory extraction → token-aware overlapping chunks → prompt → tokenizer/model"),
            ("Output", "Structured evidence summary, entities, observations, relevance, and limitations"),
            ("Loss/objective", "Pretrained seq2seq objective; no project fine-tuning or training claim"),
            ("Evaluation", "Input/output tokens, processed chunks, truncation status, and inference time"),
            ("Role", "AI-assisted evidence intelligence for human review; never a legal/forensic verdict"),
        ]
    )
    evidence_consent = processing_consent("evidence")

    source_tab, upload_tab = st.tabs(["Paste evidence text", "Upload TXT or PDF"])
    evidence_text = ""
    source_description = "Pasted text"
    with source_tab:
        pasted_text = st.text_area(
            "Evidence text",
            height=220,
            placeholder="Paste a transcript, incident narrative, extracted message log, or other text evidence…",
            key="evidence_pasted_text",
            disabled=not evidence_consent,
        )
        if evidence_consent and pasted_text.strip():
            evidence_text = pasted_text.strip()
    with upload_tab:
        evidence_upload = st.file_uploader(
            "Choose evidence",
            type=["txt", "pdf"],
            key="evidence_document_upload",
            help=f"TXT or digitally extractable PDF, up to {MAX_EVIDENCE_BYTES // (1024 * 1024)} MB.",
            disabled=not evidence_consent,
        )
        if evidence_consent and evidence_upload is not None:
            try:
                parsed = parse_evidence_file(evidence_upload.name, evidence_upload.getvalue())
                evidence_text = parsed.text
                source_description = evidence_upload.name
                page_detail = f" · {parsed.page_count} page(s)" if parsed.page_count is not None else ""
                st.success(f"Extracted {parsed.character_count:,} characters{page_detail} in memory.")
            except DocumentParseError as exc:
                st.error(str(exc))

    if evidence_text:
        with st.expander("Extracted evidence preview", expanded=True):
            st.text(evidence_text[:4_000])
            if len(evidence_text) > 4_000:
                st.caption(f"Preview shows 4,000 of {len(evidence_text):,} characters; generation uses token-aware chunks.")
        prompt = st.text_area(
            "Analyst prompt",
            value=DEFAULT_PROMPT,
            height=180,
            help="The default six-line prompt requests facts, entities, cautious observations, and limitations.",
        )
        max_new_tokens = st.slider("Maximum generated tokens", 64, 384, 192, 32)
        if st.button("Generate evidence intelligence", type="primary", disabled=not evidence_consent):
            model, error = safe_load(load_evidence_transformer, "Evidence Transformer")
            if error:
                st.error(error)
            else:
                try:
                    with st.spinner("Tokenizing evidence and generating observations…"):
                        result = model.generate(evidence_text, prompt, int(max_new_tokens))
                    st.markdown("### Generated evidence intelligence")
                    st.markdown(result.output)
                    st.caption(
                        "The Evidence Summary is generated by the pretrained Transformer. Entity and "
                        "observation indicators are conservative source-text pattern matches added to "
                        "reduce unsupported invention by the compact model."
                    )
                    if result.truncated:
                        st.warning(
                            f"Document limit reached: processed {result.chunks_processed} of "
                            f"{result.chunks_available} chunks ({result.processed_tokens:,} of "
                            f"{result.input_tokens:,} evidence tokens)."
                        )
                    metadata = st.columns(4)
                    metadata[0].metric("Input tokens", f"{result.input_tokens:,}")
                    metadata[1].metric("Processed chunks", f"{result.chunks_processed}/{result.chunks_available}")
                    metadata[2].metric("Output tokens", f"{result.output_tokens:,}")
                    metadata[3].metric("Inference time", f"{result.inference_seconds:.2f} s")
                    st.caption(
                        f"Source: {source_description} · Model: {result.model_id} · deterministic beam search"
                    )
                except (EvidenceModelError, ValueError) as exc:
                    st.error(f"Evidence generation could not continue: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected evidence-generation failure: {exc}")
    else:
        st.markdown(
            '<div class="upload-guide">Confirm authorization, then paste evidence text or upload a UTF-8 TXT/digitally extractable PDF.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="guardrail"><b>Responsible use:</b> Research/educational use only. Verify generated observations against original evidence. Generated text may contain errors and must not be the sole legal or forensic basis.</div>',
        unsafe_allow_html=True,
    )


elif page == "Model Evaluation":
    heading("Evidence-based comparison", "Model roles and measured outcomes", "Each architecture has a distinct objective. Reconstruction metrics compare output with input; they do not establish a single overall ranking or forensic validity.")
    try:
        comparison = load_image_comparison()
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            "All three rows use the same canonical 1,892-image held-out test split. "
            "Values are loaded from verified result artifacts, not validation metrics."
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Verified comparison results are unavailable: {exc}")
    roles = st.columns(3, gap="large")
    roles[0].success("**AE**\n\nUnderstandable deterministic compression and reconstruction baseline.")
    roles[1].info("**VAE V5**\n\nStrongest current reconstruction with probabilistic representation; weak prior-only generation.")
    roles[2].warning("**Transformer V2**\n\nAdds patch attention and spatial-token reconstruction; anomaly separation remains limited.")
    st.markdown('<div class="guardrail"><b>Responsible conclusion:</b> none of these models is validated as a standalone tampering detector. Never use them as the sole basis for a forensic or legal decision.</div>', unsafe_allow_html=True)
    with st.expander("Why these metrics are not interchangeable"):
        st.write("MSE, PSNR, and SSIM describe reconstruction fidelity.")
        st.write("ROC-AUC describes ranking separation; values near 0.5 are close to random ranking.")
        st.write("Different evaluation subsets must not be presented as directly equivalent.")


elif page == "Privacy & Ethical AI":
    heading(
        "Deployment safeguards",
        "Privacy-oriented and responsible-use controls",
        "Operational safeguards and GDPR-oriented design principles are documented here. Formal compliance is not claimed.",
    )
    controls = st.columns(3, gap="large")
    controls[0].markdown(
        '<div class="model-card"><span class="tag">Minimize</span><h3>Input controls</h3><p>Explicit authorization gate, allowlisted extensions, 10 MB limits, readable-content validation, and no OCR overreach.</p></div>',
        unsafe_allow_html=True,
    )
    controls[1].markdown(
        '<div class="model-card"><span class="tag">Retain briefly</span><h3>Session processing</h3><p>Uploaded content is processed in memory/session state. Project code does not automatically save evidence or generated text to repository files.</p></div>',
        unsafe_allow_html=True,
    )
    controls[2].markdown(
        '<div class="model-card"><span class="tag">Human review</span><h3>Decision boundary</h3><p>Outputs are research assistance only. They cannot establish authenticity, guilt, admissibility, or a legal conclusion.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Evidence data flow")
    st.markdown(
        '<div class="workflow">Authorized input → validated in memory → local model inference → on-screen result → explicit clear or session end</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "The pretrained language model runs locally after its public weights are downloaded; this project does not "
        "send evidence text to a paid API. On a hosted Streamlit deployment, evidence necessarily travels to that "
        "host, so HTTPS, access control, host logging, regional hosting, and organizational retention policy must be "
        "configured by the deployer."
    )

    st.markdown("### GDPR-oriented design principles")
    principles = pd.DataFrame(
        [
            {"Principle": "Data minimization", "Implemented measure": "Only the evidence needed for the selected analysis is requested; input size and type are constrained."},
            {"Principle": "Purpose limitation", "Implemented measure": "The interface states research/educational evidence analysis as the purpose and prohibits verdict claims."},
            {"Principle": "Transparency", "Implemented measure": "Model identity, pretrained status, processed-token limits, uncertainty, and output limitations are shown."},
            {"Principle": "Storage limitation", "Implemented measure": "No automatic permanent evidence save; users can clear evidence state and should end the session after use."},
            {"Principle": "Integrity/confidentiality", "Implemented measure": "Allowlisting and parsing reduce malformed inputs; production deployment still requires HTTPS, authentication, authorization, and secured logs."},
        ]
    )
    st.dataframe(principles, hide_index=True, width="stretch")
    st.warning(
        "These are privacy-oriented engineering measures and principles, not a claim of formal GDPR compliance. "
        "A production deployment requires legal review, documented lawful basis, data-subject procedures, security "
        "assessment, retention policy, and governance appropriate to its jurisdiction and use."
    )
    if st.button("Clear evidence from this app session", type="secondary"):
        cleared = clear_session_evidence()
        st.success(f"Cleared {cleared} evidence-related session value(s).")


else:
    heading(
        "Review-2 implementation",
        "Project Information",
        "A reproducible academic prototype built for Semester 7 Generative AI review and live faculty demonstration.",
    )
    details = st.columns(3)
    details[0].metric("Required architectures", "3")
    details[1].metric("Canonical test images", "1,892")
    details[2].metric("Demo samples", "24")
    st.markdown("### Implemented scope")
    st.write(
        "Quality Autoencoder V1, VAE V5, trained Vision Transformer V2, and a separate pretrained "
        "FLAN-T5 evidence-intelligence integration. GAN code/checkpoints remain preserved for the later phase."
    )
    st.markdown("### Current boundaries")
    st.write(
        "Diffusion is not implemented. No model is validated as a standalone forgery detector, and the "
        "application does not make legal, authenticity, guilt, or admissibility decisions."
    )
    st.markdown("### Runtime")
    st.code(f"Python · PyTorch {torch.__version__} · Streamlit {st.__version__} · device {DEVICE}")

st.divider()
left, right = st.columns([3, 1])
left.caption("Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation")
right.caption("CASIA v2.0 · Research use only")
