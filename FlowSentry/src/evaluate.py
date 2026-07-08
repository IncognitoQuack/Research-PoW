"""
evaluate.py
===========
Computes Accuracy, Precision, Recall, F1, False Positive Rate and ROC-AUC
for every trained model (ensemble, cnn_only, rnn_only, random_forest, svm,
signature_rule) on the held-out test split, plus confusion matrices.
Writes results/<dataset>_metrics.json and results/<dataset>_confusion_matrix.csv

Usage:
    python3 src/evaluate.py --dataset nsl_kdd --config configs/config.yaml
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)

from data_utils import load_processed
from datasets import WindowedFlowDataset
from models import build_model
from signature_baseline import SignatureRuleBaseline  # noqa: F401 (needed for joblib.load)
from utils import load_config, get_device, save_json, ensure_dirs


def fpr_from_confusion(cm: np.ndarray) -> float:
    # cm rows = true, cols = pred, binary: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def metrics_dict(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr_from_confusion(cm)),
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
    return out


@torch.no_grad()
def eval_torch_model(model, dataset, device, batch_size=512):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    for window, current, label in loader:
        window, current = window.to(device), current.to(device)
        logits = model(window, current)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()
        score = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy()
        all_preds.append(preds)
        all_labels.append(label.numpy())
        all_scores.append(score)
    return (np.concatenate(all_labels), np.concatenate(all_preds),
            np.concatenate(all_scores))


def main(dataset: str, config: dict):
    processed_root = config["paths"]["processed_dir"]
    X_train, X_test, y_train, y_test, meta = load_processed(dataset, processed_root)
    n_features = meta["n_features"]
    n_classes = len(meta["class_names"])
    device = get_device(config["latency"]["device_priority"])

    results = {}

    # ---- deep models ----
    test_ds = WindowedFlowDataset(X_test, y_test, config["data"]["window_length"])
    ckpt_dir = os.path.join(config["paths"]["checkpoints_dir"], dataset)
    for model_name in ["ensemble", "cnn_only", "rnn_only"]:
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] no checkpoint for {model_name} at {ckpt_path}")
            continue
        model = build_model(model_name, n_features, n_classes, config["model"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        y_true, y_pred, y_score = eval_torch_model(model, test_ds, device)
        results[model_name] = metrics_dict(y_true, y_pred, y_score)
        print(f"{model_name}: acc={results[model_name]['accuracy']:.4f} "
              f"fpr={results[model_name]['false_positive_rate']:.4f}")

    # ---- classical / rule baselines ----
    baseline_dir = "checkpoints/baselines"
    for name, fname in [("random_forest", "rf"), ("svm", "svm"),
                         ("signature_rule", "signature")]:
        path = os.path.join(baseline_dir, f"{dataset}_{fname}.joblib")
        if not os.path.exists(path):
            print(f"[skip] no baseline artifact for {name} at {path}")
            continue
        clf = joblib.load(path)
        y_pred = clf.predict(X_test)
        y_score = None
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
        results[name] = metrics_dict(y_test, y_pred, y_score)
        print(f"{name}: acc={results[name]['accuracy']:.4f} "
              f"fpr={results[name]['false_positive_rate']:.4f}")

    results_dir = config["paths"]["results_dir"]
    ensure_dirs(results_dir)
    save_json(results, os.path.join(results_dir, f"{dataset}_metrics.json"))

    # flat confusion-matrix CSV for convenience
    rows = []
    for model_name, m in results.items():
        cm = np.array(m["confusion_matrix"])
        rows.append({
            "model": model_name, "TN": cm[0, 0], "FP": cm[0, 1],
            "FN": cm[1, 0], "TP": cm[1, 1],
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, f"{dataset}_confusion_matrix.csv"), index=False
    )
    print(f"Saved metrics -> {results_dir}/{dataset}_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "cic_ids2017"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(args.dataset, cfg)
