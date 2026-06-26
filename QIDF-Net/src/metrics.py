"""Evaluation metrics: AUC, F1, FPR, per-class metrics, confusion matrix."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Return dict of metrics for binary (real/fake) detection."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro")),
        "f1_fake":   float(f1_score(y_true, y_pred, pos_label=1)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fpr":       float(fpr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def attribution_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> Dict:
    """Return per-class F1 and macro-average for attribution."""
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    per_class = {cls: report[cls]["f1-score"] for cls in classes if cls in report}
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
    }


def print_binary(label: str, m: Dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    for k, v in m.items():
        if k not in ("tp", "tn", "fp", "fn"):
            print(f"  {k:<14} {v:.4f}")
    print(f"  Confusion  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")


def save_results_csv(results: Dict, path: str) -> None:
    rows = []
    for split, m in results.items():
        row = {"split": split}
        row.update({k: v for k, v in m.items() if not isinstance(v, dict)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nResults saved → {path}")
