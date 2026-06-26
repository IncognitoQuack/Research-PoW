# QIDF-Net: Quantum-Inspired Dual-Feature Network for Deepfake Detection and Forensic Attribution

---

## Overview

QIDF-Net combines two complementary feature streams:

- **Classical branch** — EfficientNet-B4 extracts spatial forgery cues from RGB face crops.
- **Quantum branch** — A ZZ-FeatureMap-inspired quantum kernel is applied to DCT frequency coefficients, capturing higher-order cross-frequency correlations that GAN generators leave behind. The kernel is computed analytically (no quantum hardware required).

An entropy-weighted cross-attention fusion gate (EWCAF) adaptively combines both streams. The fused representation drives a soft-voting ensemble detector (XGBoost + RandomForest + LightGBM) and a multi-class attribution head that identifies the source generative model.

---

## Datasets

| Dataset | Access | Citation |
|---------|--------|---------|
| FaceForensics++ (c23) | [Request](https://github.com/ondyari/FaceForensics) | Rössler et al., ICCV 2019 |
| Celeb-DF v2 (cross-dataset) | [Request](https://github.com/yuezunli/celeb-deepfakeforensics) | Li et al., CVPR 2020 |

Both datasets require agreeing to a research-use licence. The download scripts in each repository handle the transfer.

A **demo dataset** (Labeled Faces in the Wild + simulated artifacts) is available for pipeline testing without the above downloads — results from the demo are not comparable to paper figures.

---

## Setup

```bash
# Python 3.10 or 3.11 recommended
pip install -r requirements.txt
```

No GPU is required (all steps run on CPU). A GPU reduces feature extraction time from ~20 min to ~2 min.

---

## Quickstart

### Option A — with FF++

```bash
# 1. Extract frames from FF++ videos
python scripts/prepare_ffpp.py \
    --ffpp_root /path/to/FaceForensics++ \
    --output_root data \
    --compression c23 \
    --frames_per_video 10 \
    --max_videos 150

# 2. Run full experiment
python run.py --config config.yaml

# 3. (Optional) cross-dataset evaluation
python run.py --config config.yaml --celeb_dir data/celeb_df_test
```

### Option B — demo pipeline test (LFW + simulated artifacts)

```bash
python scripts/demo_data.py --output_root data --n 300
python run.py --config config.yaml
```

---

## Output

```
outputs/
  results/
    results.csv      main metrics table (accuracy, AUC, F1, FPR, attribution)
    results.json     full metrics including per-class breakdown
  figures/
    fig1_architecture.pdf
    fig2_quantum_kernel.pdf
    fig3_confusion_ff++_test.pdf
    fig4_roc_curves.pdf
    fig5_ablation.pdf
    fig6_shap.pdf
    fig7_attribution_cm.pdf
    fig8_convergence.pdf
    fig9_per_class_f1.pdf
  models/
    fusion.pt        trained FusionNet weights
    detector.pkl     ensemble classifier
    attribution.pkl  attribution MLP + label encoder
  cache/             intermediate feature arrays (auto-managed)
```

---

## Configuration

All hyperparameters live in `config.yaml`.  Key settings:

| Key | Default | Effect |
|-----|---------|--------|
| `data.max_per_class` | 1500 | Cap training images per class |
| `classical.pca_components` | 256 | Reduce 1792-dim EfficientNet features |
| `quantum.n_dct_features` | 8 | DCT coefficients fed to QFK |
| `quantum.n_landmarks` | 300 | Nyström approximation points |
| `quantum.n_components` | 64 | Output dim of quantum branch |
| `fusion.epochs` | 60 | FusionNet training epochs |

---

## Reproducing Paper Results

Run with `config.yaml` unchanged on FF++ c23 with `max_videos=150` per split.  All figure PDFs are deposited directly to `outputs/figures/` and are camera-ready for LaTeX inclusion via `\includegraphics`.

---

## Licence

Code released for research use.  Datasets are governed by their respective owners' terms.
