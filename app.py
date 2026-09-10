"""Professional Streamlit interface for the Digital Evidence GenAI framework."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ae_inference import AutoencoderInference
from transformer_inference import TransformerInference
from vae_inference import VAEInference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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


@st.cache_resource(show_spinner="Loading Autoencoder…")
def load_ae():
    # Same architecture, using the improved epoch-76 checkpoint from the RTX-80 run.
    return AutoencoderInference(
        ROOT / "checkpoints" / "best_autoencoder.pth", DEVICE
    )


@st.cache_resource(show_spinner="Loading VAE V5 Final…")
def load_vae():
    return VAEInference(ROOT / "checkpoints" / "VAE_V5_FINAL.pth", DEVICE, 256)


@st.cache_resource(show_spinner="Loading Transformer V2…")
def load_transformer():
    return TransformerInference(ROOT / "checkpoints" / "transformer_v2_final.pth", DEVICE)


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
        st.markdown('<div class="small-label">Input</div>', unsafe_allow_html=True)
        st.image(original, caption="Original image", width="stretch")
    with right:
        st.markdown('<div class="small-label">Model output</div>', unsafe_allow_html=True)
        st.image(reconstructed, caption=caption, width="stretch", clamp=True)


def upload_box(key):
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key=key)
    image = uploaded_image(upload)
    if image is None:
        st.markdown('<div class="upload-guide">Upload a supported image to begin reconstruction and metric analysis.</div>', unsafe_allow_html=True)
    else:
        st.caption(f"{upload.name} · original size {image.width}×{image.height} · RGB")
    return image


with st.sidebar:
    st.markdown("## 🔬 Digital Evidence AI")
    st.caption("Semester 7 · Generative AI")
    st.divider()
    page = st.radio("Navigate", ["Project Overview", "Autoencoder", "VAE V5 Final", "Transformer V2", "Model Comparison"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Runtime device**")
    st.caption(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    st.caption("CUDA enabled" if torch.cuda.is_available() else "CUDA unavailable · CPU inference")
    st.divider()
    st.caption("CASIA v2.0 · 12,614 images")
    st.caption("Academic research prototype")

st.markdown("""<div class="hero"><div class="eyebrow">Multi-model generative AI framework</div><h1>Digital Evidence Analysis and Intelligence Generation</h1><p>Comparing deterministic, probabilistic, and attention-based image reconstruction for digital-evidence research using CASIA v2.0.</p></div>""", unsafe_allow_html=True)


if page == "Project Overview":
    heading("Project foundation", "A reconstruction-first forensic research framework", "Three complementary neural architectures reconstruct digital images, represent visual information, and expose exploratory anomaly signals. The application supports analysis—it does not issue forensic verdicts.")
    counts = st.columns(4)
    for column, label, value in zip(counts, ["CASIA images", "Authentic", "Tampered", "Held-out test"], ["12,614", "7,491", "5,123", "1,892"]):
        column.metric(label, value)
    st.markdown("### Current model modules")
    cards = st.columns(3, gap="large")
    cards[0].markdown('<div class="model-card"><span class="tag">Deterministic</span><h3>Autoencoder</h3><p>Compact-bottleneck reconstruction, compression, and denoising experiments.</p></div>', unsafe_allow_html=True)
    cards[1].markdown('<div class="model-card"><span class="tag">Probabilistic</span><h3>VAE V5 Final</h3><p>256-dimensional probabilistic representation with high-quality skip-connected reconstruction.</p></div>', unsafe_allow_html=True)
    cards[2].markdown('<div class="model-card"><span class="tag">Attention</span><h3>Transformer V2</h3><p>Patch-token reconstruction with interpretable learned attention relationships.</p></div>', unsafe_allow_html=True)
    st.markdown("### Analysis workflow")
    st.markdown('<div class="workflow">Upload → RGB preprocessing → reconstruction → MSE · PSNR · SSIM → exploratory interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="guardrail"><b>Interpretation boundary:</b> reconstruction error, thresholds, difference maps, and attention are not proof of manipulation and must not be treated as legal or forensic conclusions.</div>', unsafe_allow_html=True)
    with st.expander("Dataset safeguards and reproducibility"):
        st.write("Fixed stratified split: 70% train, 15% validation, 15% test; seed 42.")
        st.write("Ground-truth mask PNG files are excluded from normal model inputs.")
        st.write("All inputs are RGB 128×128 tensors normalized to [0,1].")


elif page == "Autoencoder":
    heading("Model 01 · deterministic reconstruction", "Convolutional Autoencoder — RTX-80 Final", "Examine reconstruction fidelity from the improved 80-epoch training run after a meaningful 24× dimensional bottleneck. Metrics compare input and reconstruction; the model does not predict a label.")
    info = st.columns(4)
    for column, label, value in zip(info, ["Input", "Latent", "Compression", "Parameters"], ["3×128×128", "32×8×8", "24×", "265,571"]):
        column.metric(label, value)
    st.markdown('<div class="workflow">RGB image → convolutional encoder → compact latent → decoder → reconstruction</div>', unsafe_allow_html=True)
    model, error = safe_load(load_ae, "Autoencoder")
    if error:
        st.error(error)
    else:
        image = upload = upload_box("ae_upload")
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
        st.write("Encoder downsamples the image; the bottleneck holds 2,048 values; the decoder restores 49,152 RGB values with Sigmoid output.")
        st.write("Lower MSE is better; higher PSNR is better; SSIM closer to 1 indicates stronger structural similarity.")
        st.write("Current checkpoint: best epoch 76 · validation MSE 0.00305005.")
        st.write("Complete held-out test: MSE 0.00303199 · PSNR 26.0159 dB · SSIM 0.792911.")


elif page == "VAE V5 Final":
    heading("Model 02 · probabilistic reconstruction", "Variational Autoencoder V5 Final", "Explore skip-connected reconstruction and a 256-dimensional probabilistic latent representation. Random prior sampling is retained as an experimental capability.")
    info = st.columns(4)
    for column, label, value in zip(info, ["Input", "Latent", "Parameters", "Checkpoint epoch"], ["3×128×128", "256", "17,599,971", "79"]):
        column.metric(label, value)
    st.markdown('<div class="workflow">Encoder → μ and log variance → latent z → skip-connected decoder → reconstruction</div>', unsafe_allow_html=True)
    model, error = safe_load(load_vae, "VAE V5 Final")
    if error:
        st.error(error)
    else:
        reconstruction_tab, generation_tab = st.tabs(["Image reconstruction", "Experimental generation"])
        with reconstruction_tab:
            image = upload_box("vae_upload")
            if image is not None:
                try:
                    with st.spinner("Reconstructing with VAE V5…"):
                        original, reconstructed, result = model.reconstruct(image)
                    compare_images(original, reconstructed, "VAE reconstruction using z = μ")
                    metrics_cards(result)
                    st.markdown('<div class="guardrail"><b>Exploratory indicator:</b> MSE is reconstruction error—not a tampering probability or standalone decision.</div>', unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"VAE reconstruction failed: {exc}")
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


elif page == "Transformer V2":
    heading("Model 03 · patch attention", "Transformer Forensic Autoencoder V2", "The image becomes 64 patch tokens. Self-attention models patch relationships while all encoded spatial tokens are retained for reconstruction.")
    info = st.columns(4)
    for column, label, value in zip(info, ["Input", "Patch size", "Patch tokens", "Embedding"], ["3×128×128", "16×16", "64", "256"]):
        column.metric(label, value)
    layers = st.columns(3)
    for column, label, value in zip(layers, ["Attention heads", "Encoder layers", "Decoder layers"], ["8", "4", "2"]):
        column.metric(label, value)
    st.markdown('<div class="workflow">Image → 8×8 patch grid → attention encoder → token decoder → reconstruction + attention view</div>', unsafe_allow_html=True)
    model, error = safe_load(load_transformer, "Transformer V2")
    if error:
        st.error(error)
    else:
        image = upload_box("transformer_upload")
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


else:
    heading("Evidence-based comparison", "Model roles and measured outcomes", "Each architecture has a distinct objective. Reconstruction metrics compare output with input; they do not establish a single overall ranking or forensic validity.")
    comparison = pd.DataFrame([
        {"Model":"Autoencoder RTX-80 Final","Primary role":"Deterministic reconstruction + compression","MSE ↓":"0.00303199","PSNR ↑":"26.0159","SSIM ↑":"0.792911","ROC-AUC":"N/A"},
        {"Model":"VAE V5 Final","Primary role":"Probabilistic representation + reconstruction","MSE ↓":"0.00070007","PSNR ↑":"32.9535","SSIM ↑":"0.961139","ROC-AUC":"0.5154"},
        {"Model":"Transformer V2","Primary role":"Patch-attention reconstruction","MSE ↓":"0.00206004*","PSNR ↑":"N/A","SSIM ↑":"N/A","ROC-AUC":"0.5455"},
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("*Transformer MSE is the saved best validation result, not the canonical test metric used for AE/VAE.")
    roles = st.columns(3, gap="large")
    roles[0].success("**AE**\n\nUnderstandable deterministic compression and reconstruction baseline.")
    roles[1].info("**VAE V5**\n\nStrongest current reconstruction with probabilistic representation; weak prior-only generation.")
    roles[2].warning("**Transformer V2**\n\nAdds patch attention and spatial-token reconstruction; anomaly separation remains limited.")
    st.markdown('<div class="guardrail"><b>Responsible conclusion:</b> none of these models is validated as a standalone tampering detector. Never use them as the sole basis for a forensic or legal decision.</div>', unsafe_allow_html=True)
    with st.expander("Why these metrics are not interchangeable"):
        st.write("MSE, PSNR, and SSIM describe reconstruction fidelity.")
        st.write("ROC-AUC describes ranking separation; values near 0.5 are close to random ranking.")
        st.write("Different evaluation subsets must not be presented as directly equivalent.")

st.divider()
left, right = st.columns([3, 1])
left.caption("Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation")
right.caption("CASIA v2.0 · Research use only")
