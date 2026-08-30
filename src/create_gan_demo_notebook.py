"""Create the Colab-ready DCGAN faculty demonstration notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text: str): return nbf.v4.new_markdown_cell(text)
def code(text: str): return nbf.v4.new_code_cell(text)


cells = [
    md("# DCGAN for CASIA Synthetic Forensic Image Generation\n\n**Generative AI course component**\n\nGenerated images are synthetic research outputs and are not genuine forensic evidence. The separate ResNet-18 real/fake classifier remains unchanged."),
    md("## 1. GAN objective\n\nTrain a Deep Convolutional GAN in which a generator converts random latent noise into 64×64 RGB images while a discriminator learns to distinguish training images from generated samples."),
    code("""from pathlib import Path
import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
PROJECT_ROOT = Path('/content/Digital-Evidence-GenAI') if IN_COLAB else Path.cwd().resolve()
if IN_COLAB and not PROJECT_ROOT.exists():
    subprocess.run(['git','clone','https://github.com/chetanraje27/Digital-Evidence-GenAI.git',str(PROJECT_ROOT)], check=True)
if not IN_COLAB:
    PROJECT_ROOT = next((p for p in [PROJECT_ROOT,*PROJECT_ROOT.parents] if (p/'src'/'dcgan.py').exists()), PROJECT_ROOT)
if IN_COLAB:
    subprocess.run([sys.executable,'-m','pip','install','-q','kagglehub','torchmetrics','torch-fidelity'], check=True)
sys.path.insert(0, str(PROJECT_ROOT/'src'))
TRAIN_GAN = False  # Change to True only for the one-time Colab T4 training run.
print('Project:', PROJECT_ROOT)
print('TRAIN_GAN:', TRAIN_GAN)"""),
    md("## 2. CASIA dataset overview\n\nCASIA v2.0 contains 12,614 RGB images: 7,491 authentic and 5,123 tampered. DCGAN training reuses only the established training manifest. Labels are not GAN targets, and ground-truth masks remain excluded."),
    code("""# Make relative manifest paths portable in Colab.
import csv
first = next(csv.DictReader(open(PROJECT_ROOT/'data/splits/train.csv', encoding='utf-8')))
expected = PROJECT_ROOT / first['image_path']
if IN_COLAB and not expected.exists():
    import kagglehub
    downloaded = Path(kagglehub.dataset_download('divg07/casia-20-image-tampering-detection-dataset'))
    casia2 = next(downloaded.rglob('CASIA2'))
    target = PROJECT_ROOT/'data/raw/CASIA2'; target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists(): os.symlink(casia2, target, target_is_directory=True)
print('Dataset ready:', expected.exists())"""),
    md("## 3. GAN preprocessing\n\nEach image is safely opened, converted to RGB, resized to 64×64, converted to a tensor, and normalized from `[0,1]` to `[-1,1]`. This matches the generator's final `Tanh` activation."),
    md("## 4. Generator architecture\n\n`z [100×1×1] → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64`\n\nConvTranspose2D blocks use BatchNorm and ReLU; the output uses Tanh. Parameters: **3,576,704**."),
    md("## 5. Discriminator architecture\n\n`3×64×64 → 64 → 128 → 256 → 512 → 1 logit`\n\nConv2D blocks use LeakyReLU(0.2), with BatchNorm except in the first block. Parameters: **2,765,568**."),
    code("""import torch
from dcgan import build_dcgan
generator, discriminator = build_dcgan(100)
print(generator)
print(discriminator)
print('Generator parameters:', sum(p.numel() for p in generator.parameters()))
print('Discriminator parameters:', sum(p.numel() for p in discriminator.parameters()))"""),
    md("## 6. Adversarial training\n\nThe discriminator minimizes binary cross-entropy for real images (target 1) and detached generated images (target 0). The generator then tries to make the discriminator classify generated images as real (target 1). `BCEWithLogitsLoss` is used because the discriminator returns logits."),
    md("## 7. Training configuration\n\n| Setting | Value |\n|---|---:|\n| Latent dimension | 100 |\n| Batch size | 64 |\n| Epochs | 30 |\n| Learning rate | 0.0002 |\n| Adam betas | (0.5, 0.999) |\n| Sample interval | 5 epochs |"),
    code("""from argparse import Namespace
if TRAIN_GAN:
    from train_dcgan import train
    train_args = Namespace(
        splits_dir=PROJECT_ROOT/'data/splits',
        generator_checkpoint=PROJECT_ROOT/'checkpoints/best_generator.pth',
        discriminator_checkpoint=PROJECT_ROOT/'checkpoints/best_discriminator.pth',
        history_path=PROJECT_ROOT/'results/gan_training_history.csv',
        summary_path=PROJECT_ROOT/'results/gan_training_summary.json',
        curve_path=PROJECT_ROOT/'outputs/gan/gan_loss_curve.png',
        samples_dir=PROJECT_ROOT/'outputs/gan',
        final_samples_path=PROJECT_ROOT/'outputs/gan/final_generated_samples.png',
        image_size=64, latent_dim=100, batch_size=64, num_workers=2,
        epochs=30, learning_rate=.0002, beta1=.5, beta2=.999,
        sample_interval=5, seed=42, smoke_test=False, smoke_batches=2)
    training_summary = train(train_args)
else:
    import json
    summary_path = PROJECT_ROOT/'results/gan_training_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('GAN results are absent. Set TRAIN_GAN=True for the one-time Colab GPU run.')
    training_summary = json.loads(summary_path.read_text())
training_summary"""),
    code("""if TRAIN_GAN:
    from evaluate_dcgan import evaluate
    evaluation = evaluate(Namespace(
        splits_dir=PROJECT_ROOT/'data/splits',
        generator_checkpoint=PROJECT_ROOT/'checkpoints/best_generator.pth',
        metrics_path=PROJECT_ROOT/'results/gan_test_metrics.json',
        image_size=64, latent_dim=100, batch_size=64, num_workers=2,
        seed=42, expected_test_count=1892, inception_splits=10))
else:
    evaluation = json.loads((PROJECT_ROOT/'results/gan_test_metrics.json').read_text())
evaluation"""),
    md("## 8. Generator and discriminator loss curves"),
    code("""from IPython.display import display, Image as DisplayImage
display(DisplayImage(filename=str(PROJECT_ROOT/'outputs/gan/gan_loss_curve.png')))"""),
    md("## 9. Generated samples over epochs\n\nFixed latent noise makes visual changes across epochs comparable."),
    code("""for epoch in (5,10,15,20,25,30):
    path = PROJECT_ROOT/'outputs/gan'/f'generated_epoch_{epoch:02d}.png'
    if path.exists():
        print('Epoch', epoch); display(DisplayImage(filename=str(path)))"""),
    md("## 10. Final generated images\n\nThese are synthetic DCGAN outputs, not real CASIA evidence."),
    code("display(DisplayImage(filename=str(PROJECT_ROOT/'outputs/gan/final_generated_samples.png')))"),
    md("## 11. FID\n\nFID compares Inception feature distributions of all 1,892 real test images and 1,892 separately generated images. Lower is better."),
    code("print('FID:', evaluation['fid'])"),
    md("## 12. Inception Score\n\nInception Score is reported when the validated implementation is available. Higher scores indicate confident and varied class predictions, but the metric is imperfect for forensic-domain images."),
    code("print('Inception Score:', evaluation.get('inception_score_mean')); print('IS std:', evaluation.get('inception_score_std')); print('Warning:', evaluation.get('warning'))"),
    md("## 13. Limitations and ethical note\n\n- GAN training can be unstable and losses do not directly measure image quality.\n- Mode collapse may reduce diversity.\n- CASIA is a forensic dataset, not a general natural-image generation benchmark.\n- FID and Inception Score use ImageNet features and have domain limitations.\n- Generated images are synthetic research outputs and must never be presented as genuine forensic evidence."),
    code("""print('GPU:', training_summary['gpu'])
print('Epochs completed:', training_summary['epochs_completed'])
print('Final generator loss:', training_summary['final_generator_loss'])
print('Final discriminator loss:', training_summary['final_discriminator_loss'])
print('FID:', evaluation['fid'])
print('Inception Score:', evaluation.get('inception_score_mean'))
print('Training time:', training_summary['training_time_seconds'])
print('Generator checkpoint:', training_summary['generator_checkpoint'])
print('Discriminator checkpoint:', training_summary['discriminator_checkpoint'])
print('Errors/warnings:', evaluation.get('warning'))"""),
    code("""# In Colab, package all trained GAN artifacts for download/preservation.
if IN_COLAB and TRAIN_GAN:
    import shutil
    archive = shutil.make_archive('/content/gan_casia_artifacts','zip',PROJECT_ROOT,
        base_dir='outputs/gan')
    print('Output archive:', archive)
    print('Also download checkpoints/ and results/ before ending the runtime.')"""),
]

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"name": "GAN_CASIA_Complete_Demo.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(notebook, ROOT / "notebooks" / "GAN_CASIA_Complete_Demo.ipynb")
print("Created GAN_CASIA_Complete_Demo.ipynb")
