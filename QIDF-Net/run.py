"""
QIDF-Net — Quantum-Inspired Dual-Feature Network for Deepfake Detection
and Forensic Attribution

Main experiment runner.  Runs the complete pipeline:
  1. Load data splits (real/fake image folders)
  2. Extract EfficientNet-B4 classical features
  3. Extract DCT frequency features
  4. Fit Quantum Frequency Kernel projector (Nyström)
  5. Train FusionNet (detection + attribution)
  6. Extract fused representations
  7. Train ensemble detector (XGB + RF + LGB)
  8. Train attribution MLP
  9. Evaluate on test set and (optionally) Celeb-DF
  10. Run ablation study
  11. Generate all paper figures
  12. Save results CSV

Usage:
  python run.py --config config.yaml
  python run.py --config config.yaml --celeb_dir data/celeb_df_test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import shap

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.utils       import load_config, set_seed, get_device, get_logger, ensure_dirs
from src.dataset     import load_split, split_summary
from src.classical   import extract_classical_features
from src.dct_features import extract_dct_features
from src.quantum_kernel import QuantumKernelProjector, extract_quantum_features, zz_kernel_matrix
from src.fusion      import train_fusion, extract_fused_features, FusionNet
from src.classifiers import (train_detector, predict_detector,
                              train_attribution, predict_attribution)
from src.metrics     import (binary_metrics, attribution_metrics,
                              print_binary, save_results_csv)
from src.visualization import (
    fig_architecture, fig_quantum_kernel, fig_confusion_detection,
    fig_roc_curves, fig_ablation, fig_shap_summary,
    fig_attribution_cm, fig_convergence, fig_per_class_f1,
)

log = get_logger("qidf_net")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_le(att_classes):
    le = LabelEncoder()
    le.classes_ = np.array(att_classes)
    return le


def _pca_reduce(X_train, X_val, X_test, n_comp, seed):
    pca = PCA(n_components=n_comp, random_state=seed)
    Xt = pca.fit_transform(X_train)
    Xv = pca.transform(X_val)
    Xs = pca.transform(X_test)
    return Xt, Xv, Xs, pca


def _scale(X_train, X_val, X_test):
    sc = StandardScaler()
    return (sc.fit_transform(X_train),
            sc.transform(X_val),
            sc.transform(X_test),
            sc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg_path: str, celeb_dir: str = "") -> None:
    t0 = time.time()
    cfg = load_config(cfg_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    log.info("Device: %s", device)

    ensure_dirs(
        cfg["output"]["cache_dir"],
        cfg["output"]["models_dir"],
        cfg["output"]["results_dir"],
        cfg["output"]["figures_dir"],
    )

    # ── 1. Load splits ──────────────────────────────────────────────────────
    log.info("Loading data splits …")
    att_classes = cfg["attribution"]["classes"]

    tr_paths, tr_labels, tr_names = load_split(
        cfg["data"]["train_dir"], cfg["data"]["max_per_class"], seed)
    va_paths, va_labels, va_names = load_split(
        cfg["data"]["val_dir"], cfg["data"]["max_per_class"] // 3, seed)
    te_paths, te_labels, te_names = load_split(
        cfg["data"]["test_dir"], None, seed)

    log.info("Train %s", split_summary(tr_paths, tr_labels, tr_names))
    log.info("Val   %s", split_summary(va_paths, va_labels, va_names))
    log.info("Test  %s", split_summary(te_paths, te_labels, te_names))

    # Build attribution labels (for fake samples only; real → "Real")
    tr_att = [n if l == 1 else "Real" for n, l in zip(tr_names, tr_labels)]
    va_att = [n if l == 1 else "Real" for n, l in zip(va_names, va_labels)]
    te_att = [n if l == 1 else "Real" for n, l in zip(te_names, te_labels)]

    # Clamp attribution labels to known classes
    known = set(att_classes)
    tr_att = [a if a in known else "Fake" for a in tr_att]
    va_att = [a if a in known else "Fake" for a in va_att]
    te_att = [a if a in known else "Fake" for a in te_att]

    # Use whatever classes actually appear
    actual_att_classes = sorted(set(tr_att))

    # ── 2. Classical features ───────────────────────────────────────────────
    log.info("\n[Step 2] EfficientNet-B4 feature extraction")
    Fc_tr = extract_classical_features(tr_paths, device, cfg["classical"]["batch_size"],
                                       cfg["output"]["cache_dir"], "train")
    Fc_va = extract_classical_features(va_paths, device, cfg["classical"]["batch_size"],
                                       cfg["output"]["cache_dir"], "val")
    Fc_te = extract_classical_features(te_paths, device, cfg["classical"]["batch_size"],
                                       cfg["output"]["cache_dir"], "test")

    # PCA reduce classical features
    n_pca = cfg["classical"]["pca_components"]
    Fc_tr, Fc_va, Fc_te, _ = _pca_reduce(Fc_tr, Fc_va, Fc_te, n_pca, seed)
    log.info("Classical features: %s", Fc_tr.shape)

    # ── 3. DCT features ─────────────────────────────────────────────────────
    log.info("\n[Step 3] DCT frequency feature extraction")
    n_dct = cfg["quantum"]["n_dct_features"]
    Fd_tr = extract_dct_features(tr_paths, n_dct, cfg["output"]["cache_dir"], "train")
    Fd_va = extract_dct_features(va_paths, n_dct, cfg["output"]["cache_dir"], "val")
    Fd_te = extract_dct_features(te_paths, n_dct, cfg["output"]["cache_dir"], "test")

    # ── 4. Quantum kernel features ──────────────────────────────────────────
    log.info("\n[Step 4] Quantum Frequency Kernel projection")
    qcfg = cfg["quantum"]
    Fq_tr, qkp = extract_quantum_features(
        Fd_tr, None, "train",
        cfg["output"]["cache_dir"], qcfg["n_landmarks"], qcfg["n_components"], seed,
    )
    Fq_va, _   = extract_quantum_features(
        Fd_va, qkp, "val",
        cfg["output"]["cache_dir"], qcfg["n_landmarks"], qcfg["n_components"], seed,
    )
    Fq_te, _   = extract_quantum_features(
        Fd_te, qkp, "test",
        cfg["output"]["cache_dir"], qcfg["n_landmarks"], qcfg["n_components"], seed,
    )
    log.info("Quantum features: %s", Fq_tr.shape)

    # ── 5. Train FusionNet ──────────────────────────────────────────────────
    log.info("\n[Step 5] Training FusionNet (detection + attribution)")
    le = _make_le(actual_att_classes)
    fusion_model, le = train_fusion(
        Fc_tr, Fq_tr, tr_labels, tr_att,
        Fc_va, Fq_va, va_labels, va_att,
        actual_att_classes,
        cfg["fusion"], device,
        save_path=os.path.join(cfg["output"]["models_dir"], "fusion.pt"),
    )

    # ── 6. Fused representations ────────────────────────────────────────────
    log.info("\n[Step 6] Extracting fused representations")
    Ff_tr = extract_fused_features(fusion_model, Fc_tr, Fq_tr, device)
    Ff_va = extract_fused_features(fusion_model, Fc_va, Fq_va, device)
    Ff_te = extract_fused_features(fusion_model, Fc_te, Fq_te, device)
    log.info("Fused features: %s", Ff_tr.shape)

    # ── 7. Ensemble detector ────────────────────────────────────────────────
    log.info("\n[Step 7] Training ensemble detector")
    detector = train_detector(
        Ff_tr, tr_labels, cfg["ensemble"],
        save_path=os.path.join(cfg["output"]["models_dir"], "detector.pkl"),
        seed=seed,
    )

    # ── 8. Attribution ──────────────────────────────────────────────────────
    log.info("\n[Step 8] Training attribution MLP")
    # Use only fake samples for attribution training
    mask_tr_fake = tr_labels == 1
    att_model, le = train_attribution(
        Ff_tr[mask_tr_fake],
        [tr_att[i] for i, m in enumerate(mask_tr_fake) if m],
        actual_att_classes,
        cfg["attribution"], le,
        save_path=os.path.join(cfg["output"]["models_dir"], "attribution.pkl"),
        seed=seed,
    )

    # ── 9. Evaluation ───────────────────────────────────────────────────────
    log.info("\n[Step 9] Evaluation")
    results = {}

    def evaluate_split(Ff, labels, att_names, tag, paths=None):
        preds, proba = predict_detector(detector, Ff)
        m = binary_metrics(labels, preds, proba)
        print_binary(tag, m)

        mask_fake = labels == 1
        Ff_fake = Ff[mask_fake]
        att_true = [att_names[i] for i, mk in enumerate(mask_fake) if mk]
        if len(Ff_fake) > 0:
            att_pred, _ = predict_attribution(att_model, le, Ff_fake)
            att_m = attribution_metrics(att_true, list(att_pred), actual_att_classes)
            m.update({"att_macro_f1": att_m["macro_f1"], "att_acc": att_m["accuracy"]})
            m["att_per_class"] = att_m["per_class"]
            print(f"  Attribution macro-F1={att_m['macro_f1']:.4f}  acc={att_m['accuracy']:.4f}")
        else:
            att_pred = []

        results[tag] = m
        return preds, proba, list(att_pred)

    tr_preds, tr_proba, _ = evaluate_split(Ff_tr, tr_labels, tr_att, "Train")
    va_preds, va_proba, _ = evaluate_split(Ff_va, va_labels, va_att, "Val")
    te_preds, te_proba, te_att_pred = evaluate_split(Ff_te, te_labels, te_att, "Test (FF++)")

    # Optional: cross-dataset on Celeb-DF
    celeb_dir = celeb_dir or cfg["data"].get("celeb_dir", "")
    if celeb_dir and Path(celeb_dir).is_dir():
        log.info("Evaluating on Celeb-DF …")
        cb_paths, cb_labels, cb_names = load_split(celeb_dir, None, seed)
        Fc_cb = extract_classical_features(cb_paths, device, cfg["classical"]["batch_size"],
                                           cfg["output"]["cache_dir"], "celeb")
        Fc_cb, *_ = (PCA(n_components=n_pca, random_state=seed)
                     .fit(Fc_tr).transform(Fc_cb),)  # project using train PCA
        Fd_cb = extract_dct_features(cb_paths, n_dct, cfg["output"]["cache_dir"], "celeb")
        Fq_cb, _ = extract_quantum_features(Fd_cb, qkp, "celeb",
                                            cfg["output"]["cache_dir"],
                                            qcfg["n_landmarks"], qcfg["n_components"], seed)
        Ff_cb = extract_fused_features(fusion_model, Fc_cb, Fq_cb, device)
        cb_att = [n if l == 1 else "Real" for n, l in zip(cb_names, cb_labels)]
        evaluate_split(Ff_cb, cb_labels, cb_att, "Test (Celeb-DF)")

    # Save results
    csv_path = os.path.join(cfg["output"]["results_dir"], "results.csv")
    save_results_csv(results, csv_path)

    # Also dump full JSON
    json_path = os.path.join(cfg["output"]["results_dir"], "results.json")
    _results_clean = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(_results_clean, f, indent=2)

    # ── 10. Ablation ────────────────────────────────────────────────────────
    log.info("\n[Step 10] Ablation study")
    ablation_data = _run_ablation(
        Fc_tr, Fq_tr, tr_labels,
        Fc_te, Fq_te, te_labels,
        cfg, seed, device, fusion_model,
        Ff_tr, Ff_te,
    )

    # ── 11. Figures ─────────────────────────────────────────────────────────
    log.info("\n[Step 11] Generating figures")
    figs = cfg["output"]["figures_dir"]

    fig_architecture(figs)

    # Quantum kernel sample (small subset of training kernel)
    K_sample = zz_kernel_matrix(Fd_tr[:80], Fd_tr[:80])
    fig_quantum_kernel(K_sample, figs)

    # Confusion matrix — binary detection
    fig_confusion_detection(
        te_labels, te_preds, ["Real", "Fake"], "FF++ Test", figs
    )

    # ROC curves (using ablation variants as baselines)
    roc_data = {}
    for method, res in ablation_data.items():
        roc_data[method] = {
            "y_true": te_labels,
            "y_prob": res["proba"],
        }
    color_list = ["#2166AC", "#D6604D", "#F4A582", "#92C5DE", "#4DAC26", "#E9A3C9"]
    for i, k in enumerate(roc_data):
        roc_data[k]["color"] = color_list[i % len(color_list)]
    fig_roc_curves(roc_data, figs)

    # Ablation bar chart
    abl_chart = {k: {"AUC": v["auc"], "F1": v["f1_macro"]}
                 for k, v in ablation_data.items()}
    fig_ablation(abl_chart, figs)

    # SHAP (on XGBoost base estimator)
    try:
        xgb_clf = detector.named_estimators_["xgb"]
        explainer = shap.TreeExplainer(xgb_clf)
        shap_vals = explainer.shap_values(Ff_te[:200])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        feat_names = [f"F{i}" for i in range(Ff_te.shape[1])]
        fig_shap_summary(shap_vals, feat_names, figs)
    except Exception as e:
        log.warning("SHAP figure skipped: %s", e)

    # Attribution confusion matrix
    if te_att_pred and results.get("Test (FF++)"):
        mask_fake = te_labels == 1
        att_true_fake = [te_att[i] for i, m in enumerate(mask_fake) if m]
        if att_true_fake and te_att_pred:
            fig_attribution_cm(att_true_fake, te_att_pred, actual_att_classes, figs)

    # Convergence — placeholder from a monitored re-run
    # (FusionNet doesn't expose loss history by default; use dummy if needed)
    log.info("Figures saved to: %s", figs)

    elapsed = time.time() - t0
    log.info("\n✓ All done in %.1f s (%.1f min)", elapsed, elapsed / 60)
    log.info("Results: %s", cfg["output"]["results_dir"])
    log.info("Figures: %s", figs)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation(Fc_tr, Fq_tr, y_tr, Fc_te, Fq_te, y_te,
                  cfg, seed, device, full_fusion_model,
                  Ff_tr_full, Ff_te_full):
    """
    Compare five configurations on test set:
      (a) Classical only (EfficientNet + XGB)
      (b) Quantum only   (QFK + XGB)
      (c) Concat + Ensemble (no learned fusion)
      (d) Fusion (static weights) + Ensemble
      (e) QIDF-Net full (EWCAF + Ensemble)
    """
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    results = {}

    def _quick_xgb(X_tr, y_tr, X_te, y_te, label):
        clf = XGBClassifier(
            n_estimators=100, max_depth=5,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        m = binary_metrics(y_te, preds, proba)
        m["proba"] = proba
        log.info("  Ablation [%s]  AUC=%.4f  F1=%.4f", label, m["auc"], m["f1_macro"])
        return m

    # (a) Classical only
    results["Classical only"] = _quick_xgb(Fc_tr, y_tr, Fc_te, y_te, "classical")

    # (b) Quantum only
    results["Quantum only"] = _quick_xgb(Fq_tr, y_tr, Fq_te, y_te, "quantum")

    # (c) Concat (no fusion)
    Fc_tr_s = StandardScaler().fit_transform(Fc_tr)
    Fq_tr_s = StandardScaler().fit_transform(Fq_tr)
    Fc_te_s = StandardScaler().fit_transform(Fc_te)
    Fq_te_s = StandardScaler().fit_transform(Fq_te)
    X_concat_tr = np.hstack([Fc_tr_s, Fq_tr_s])
    X_concat_te = np.hstack([Fc_te_s, Fq_te_s])
    results["Concat + Ensemble"] = _quick_xgb(X_concat_tr, y_tr, X_concat_te, y_te, "concat")

    # (d) Static fusion (equal weights, no entropy gate)
    # Reuse FusionNet but zero out the gate gradient effect — approximate by
    # averaging the two branch projections manually
    Fc_tr_t = torch.tensor(Fc_tr, dtype=torch.float32)
    Fq_tr_t = torch.tensor(Fq_tr, dtype=torch.float32)
    Fc_te_t = torch.tensor(Fc_te, dtype=torch.float32)
    Fq_te_t = torch.tensor(Fq_te, dtype=torch.float32)

    # Project with static 0.5/0.5 weight (approximate)
    with torch.no_grad():
        c_tr = full_fusion_model.proj_c(Fc_tr_t).numpy()
        q_tr = full_fusion_model.proj_q(Fq_tr_t).numpy()
        c_te = full_fusion_model.proj_c(Fc_te_t).numpy()
        q_te = full_fusion_model.proj_q(Fq_te_t).numpy()
    static_tr = 0.5 * c_tr + 0.5 * q_tr
    static_te = 0.5 * c_te + 0.5 * q_te
    results["Static fusion"] = _quick_xgb(static_tr, y_tr, static_te, y_te, "static-fusion")

    # (e) Full QIDF-Net
    from src.classifiers import train_detector, predict_detector
    det_full = train_detector(Ff_tr_full, y_tr, cfg["ensemble"], seed=seed,
                              save_path=os.path.join(cfg["output"]["models_dir"],
                                                     "detector_ablation_full.pkl"))
    preds_f, proba_f = predict_detector(det_full, Ff_te_full)
    m_full = binary_metrics(y_te, preds_f, proba_f)
    m_full["proba"] = proba_f
    log.info("  Ablation [QIDF-Net Full]  AUC=%.4f  F1=%.4f",
             m_full["auc"], m_full["f1_macro"])
    results["QIDF-Net (Ours)"] = m_full

    return results


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QIDF-Net experiment runner")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--celeb_dir", default="",
                        help="Path to Celeb-DF test split (optional)")
    args = parser.parse_args()
    main(args.config, args.celeb_dir)
