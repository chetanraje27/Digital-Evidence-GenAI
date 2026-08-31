"""Streamlit GUI for the Digital Evidence Generative AI Framework."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ae_inference import AutoencoderInference
from gan_inference import GANInference
from vae_inference import VAEInference


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@st.cache_resource(show_spinner="Loading Autoencoder checkpoint…")
def load_ae() -> AutoencoderInference:
    return AutoencoderInference(ROOT / "checkpoints" / "best_autoencoder.pth", DEVICE)


@st.cache_resource(show_spinner="Loading VAE V2 checkpoint…")
def load_vae() -> VAEInference:
    return VAEInference(ROOT / "checkpoints" / "best_vae_v2.pth", DEVICE, 128)


@st.cache_resource(show_spinner="Loading GAN checkpoints…")
def load_gan() -> GANInference:
    return GANInference(
        ROOT / "checkpoints" / "best_generator.pth",
        ROOT / "checkpoints" / "best_discriminator.pth", DEVICE, 100,
    )


def safe_load(loader, label: str):
    try:
        return loader(), None
    except FileNotFoundError as exc:
        return None, f"{label} checkpoint is missing. {exc}"
    except Exception as exc:
        return None, f"{label} could not be loaded: {exc}"


def uploaded_image(uploaded) -> Image.Image | None:
    if uploaded is None: return None
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        st.error(f"Unsupported file type: {suffix or 'unknown'}. Upload JPG, JPEG, PNG, BMP, TIF, or TIFF.")
        return None
    try:
        image = Image.open(uploaded)
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"The uploaded image could not be read: {exc}")
        return None


def metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(3)
    columns[0].metric("MSE ↓", f"{metrics['mse']:.6f}")
    columns[1].metric("PSNR ↑", f"{metrics['psnr']:.3f} dB")
    columns[2].metric("SSIM ↑", f"{metrics['ssim']:.4f}")


st.set_page_config(page_title="Digital Evidence Generative AI Framework", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding:1.3rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#102a43,#176b87);color:white;margin-bottom:1.2rem;}
.hero h1 {margin:0;font-size:2.25rem}.hero p {margin:.35rem 0 0;color:#d9edf2;font-size:1.05rem}
.model-card {border:1px solid #d7e1e8;border-radius:13px;padding:1rem;min-height:125px;background:#f8fbfc;}
.notice {border-left:4px solid #176b87;padding:.65rem .9rem;background:#eff8fa;border-radius:6px;}
</style>
<div class="hero"><h1>Digital Evidence Generative AI Framework</h1>
<p>Autoencoder • Variational Autoencoder • GAN</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Framework")
    st.caption("Multi-Model Generative AI Framework for Digital Evidence Analysis and Intelligence Generation")
    st.divider()
    st.write("**Runtime device**")
    st.code(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    st.info("Generated images are synthetic research outputs, not genuine forensic evidence.")

overview_tab, ae_tab, vae_tab, gan_tab, comparison_tab = st.tabs([
    "Project Overview", "Autoencoder", "VAE", "GAN", "Model Comparison"
])

with overview_tab:
    st.subheader("CASIA v2.0 Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total images", "12,614"); c2.metric("Authentic", "7,491"); c3.metric("Tampered", "5,123")
    st.markdown("This system demonstrates complementary generative AI techniques for digital image evidence analysis, reconstruction, compression, probabilistic representation, and synthetic generation.")
    cards = st.columns(3)
    cards[0].markdown('<div class="model-card"><h3>Autoencoder</h3><p>Image reconstruction and 24× latent compression.</p></div>', unsafe_allow_html=True)
    cards[1].markdown('<div class="model-card"><h3>Variational Autoencoder</h3><p>Probabilistic reconstruction and synthetic generation.</p></div>', unsafe_allow_html=True)
    cards[2].markdown('<div class="model-card"><h3>DCGAN</h3><p>Adversarial synthetic forensic-image generation.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice">Authentic/tampered labels support exploratory comparisons only. Reconstruction error alone does not prove forgery.</div>', unsafe_allow_html=True)

with ae_tab:
    st.subheader("Autoencoder Reconstruction")
    st.caption("Preprocessing: RGB → 128×128 → tensor in [0,1]")
    ae, ae_error = safe_load(load_ae, "Autoencoder")
    if ae_error: st.error(ae_error)
    else:
        info = st.columns(2); info[0].metric("Compression ratio", "24×"); info[1].metric("Parameters", "265,571")
        upload = st.file_uploader("Upload an image for AE reconstruction", type=["jpg","jpeg","png","bmp","tif","tiff"], key="ae_upload")
        image = uploaded_image(upload)
        if image is not None:
            try:
                original, reconstructed, metrics = ae.reconstruct(image)
                left, right = st.columns(2)
                left.image(original, caption="Original image", width="stretch")
                right.image(reconstructed, caption="Reconstructed image", width="stretch", clamp=True)
                metric_cards(metrics)
                with st.expander("How to read these metrics"):
                    st.write("Lower MSE is better. Higher PSNR is better. SSIM closer to 1 is better.")
                    st.warning("These values do not classify an image as authentic or tampered.")
            except Exception as exc: st.error(f"AE reconstruction failed: {exc}")

with vae_tab:
    st.subheader("Variational Autoencoder V2")
    st.caption("Probabilistic latent reconstruction and generation • RGB 128×128 • [0,1]")
    vae, vae_error = safe_load(load_vae, "VAE V2")
    if vae_error: st.error(vae_error)
    else:
        st.metric("Latent dimension", "128")
        upload = st.file_uploader("Upload an image for VAE reconstruction", type=["jpg","jpeg","png","bmp","tif","tiff"], key="vae_upload")
        image = uploaded_image(upload)
        if image is not None:
            try:
                original, reconstructed, metrics = vae.reconstruct(image)
                left, right = st.columns(2)
                left.image(original, caption="Original", width="stretch")
                right.image(reconstructed, caption="VAE reconstruction (z = μ)", width="stretch", clamp=True)
                metric_cards(metrics)
                st.caption("Reconstruction statistics are exploratory and are not forgery predictions.")
            except Exception as exc: st.error(f"VAE reconstruction failed: {exc}")
        st.divider()
        if st.button("Generate Synthetic Image", type="primary", key="vae_generate"):
            try:
                st.image(vae.generate(), caption="Synthetic VAE-generated image", width=420, clamp=True)
                st.warning("Synthetic research output — not real forensic evidence.")
            except Exception as exc: st.error(f"VAE generation failed: {exc}")

with gan_tab:
    st.subheader("DCGAN Synthetic Generation")
    st.caption("No uploaded image is required. Latent noise is sampled from N(0,I).")
    gan, gan_error = safe_load(load_gan, "GAN")
    if gan_error: st.error(gan_error)
    else:
        model_info = st.columns(3)
        model_info[0].metric("Latent dimension", "100")
        model_info[1].metric("Generator parameters", "3,576,704")
        model_info[2].metric("Discriminator parameters", "2,765,568")
        quality = st.columns(2); quality[0].metric("Final FID ↓", "169.88"); quality[1].metric("Inception Score ↑", "2.95 ± 0.13")
        count = st.slider("Number of samples", 1, 8, 1)
        if st.button("Generate Synthetic Evidence Image", type="primary", key="gan_generate"):
            try:
                samples = gan.generate(count)
                columns = st.columns(min(count, 4))
                for index, sample in enumerate(samples):
                    columns[index % len(columns)].image(sample, caption=f"Synthetic GAN image {index + 1}", width="stretch", clamp=True)
                st.warning("Synthetic GAN-generated research images — not genuine forensic evidence.")
            except Exception as exc: st.error(f"GAN generation failed: {exc}")

with comparison_tab:
    st.subheader("Model Comparison")
    comparison = pd.DataFrame([
        {"Model":"Autoencoder", "Purpose":"Reconstruction + compression", "MSE":"0.00413435", "PSNR":"24.5603", "SSIM":"0.725357", "FID":"—", "Inception Score":"—"},
        {"Model":"VAE V2", "Purpose":"Probabilistic reconstruction + generation", "MSE":"0.0281680", "PSNR":"15.9348", "SSIM":"0.290189", "FID":"321.358", "Inception Score":"—"},
        {"Model":"DCGAN", "Purpose":"Synthetic image generation", "MSE":"—", "PSNR":"—", "SSIM":"—", "FID":"169.88", "Inception Score":"2.95"},
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.info("These metrics measure different model objectives and should not all be compared directly.")
    with st.expander("Interpretation"):
        st.write("AE/VAE reconstruction metrics compare an output with its input. FID compares real and generated feature distributions. Inception Score evaluates generated-image confidence and diversity using an external classifier.")

st.divider()
st.caption("Semester 7 Generative AI Project • CASIA v2.0 • Faculty demonstration interface")
