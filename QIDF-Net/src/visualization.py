"""
Paper figure generation — all nine figures used in the manuscript.

All plots are saved as high-DPI PDFs suitable for Elsevier/LaTeX submission.
Style: clean white background, seaborn-white grid, no decorative clutter.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Global aesthetics
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

PALETTE = {
    "proposed": "#2166AC",
    "baseline1": "#D6604D",
    "baseline2": "#F4A582",
    "baseline3": "#92C5DE",
    "baseline4": "#4DAC26",
    "baseline5": "#E9A3C9",
    "baseline6": "#762A83",
}


def _savefig(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Fig 1 — Architecture diagram
# ---------------------------------------------------------------------------

def fig_architecture(out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=9, sub=None):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=0.8, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                    ha="center", va="center", fontsize=7.5, style="italic", zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))

    # Input
    box(0.1, 2.4, 1.5, 1.2, "Input Face", "#F7F7F7", sub="224×224 RGB")

    # Classical branch
    box(2.0, 4.0, 2.2, 1.0, "EfficientNet-B4", "#AEC6E8", sub="1792-dim features")
    box(2.0, 2.8, 2.2, 0.9, "PCA reduction", "#AEC6E8", sub="→ 256-dim")

    # Quantum branch
    box(2.0, 1.4, 2.2, 1.0, "DCT (8×8 blocks)", "#D4EDDA", sub="top-8 AC coeffs")
    box(2.0, 0.2, 2.2, 0.9, "QFK + Nyström", "#D4EDDA", sub="→ 64-dim")

    # EWCAF
    box(4.8, 1.8, 2.4, 1.4, "EWCAF", "#FDDC9A", fontsize=10, sub="Entropy-Weighted\nCross-Attention")

    # Fused
    box(8.0, 1.8, 1.8, 1.4, "Fused Rep.", "#E8DAEF", sub="128-dim")

    # Heads
    box(10.2, 3.2, 2.5, 1.0, "Detection Head", "#FADADD", sub="XGB+RF+LGB ensemble")
    box(10.2, 2.0, 2.5, 0.9, "Attribution Head", "#C9E1F5", sub="MLP (5-class)")
    box(10.2, 0.8, 2.5, 0.9, "XAI Layer", "#FAD7AC", sub="SHAP + Grad-CAM")

    # Arrows
    arrow(1.6, 3.0, 2.0, 4.5)
    arrow(1.6, 3.0, 2.0, 1.9)
    arrow(4.2, 4.5, 4.8, 2.85)
    arrow(4.2, 2.8, 4.8, 2.5)
    arrow(4.2, 0.65, 4.8, 2.1)
    arrow(4.2, 1.9, 4.8, 2.2)
    arrow(7.2, 2.5, 8.0, 2.5)
    arrow(9.8, 3.0, 10.2, 3.7)
    arrow(9.8, 2.5, 10.2, 2.45)
    arrow(9.8, 2.1, 10.2, 1.25)

    ax.set_title("QIDF-Net System Architecture", fontsize=13, pad=10)
    _savefig(fig, os.path.join(out_dir, "fig1_architecture.pdf"))


# ---------------------------------------------------------------------------
# Fig 2 — Quantum kernel heatmap
# ---------------------------------------------------------------------------

def fig_quantum_kernel(K_sample: np.ndarray, out_dir: str) -> None:
    """Visualise a portion of the quantum kernel matrix K_Q."""
    n = min(60, K_sample.shape[0])
    K_vis = K_sample[:n, :n]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K_vis, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="$K_Q(x_i, x_j)$")
    ax.set_title("Quantum Frequency Kernel matrix $K_Q$ (sample subset)", pad=8)
    ax.set_xlabel("Sample index $j$")
    ax.set_ylabel("Sample index $i$")
    _savefig(fig, os.path.join(out_dir, "fig2_quantum_kernel.pdf"))


# ---------------------------------------------------------------------------
# Fig 3 — Confusion matrices (detection per manipulation type)
# ---------------------------------------------------------------------------

def fig_confusion_detection(y_true, y_pred, class_names, split_label, out_dir) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {split_label}", pad=8)
    plt.xticks(rotation=30, ha="right")
    _savefig(fig, os.path.join(out_dir, f"fig3_confusion_{split_label.lower().replace(' ', '_')}.pdf"))


# ---------------------------------------------------------------------------
# Fig 4 — ROC curves (all methods)
# ---------------------------------------------------------------------------

def fig_roc_curves(roc_data: Dict[str, dict], out_dir: str) -> None:
    """
    roc_data: {method_name: {"y_true": ..., "y_prob": ..., "color": ...}}
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    for name, d in roc_data.items():
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        auc_val = auc(fpr, tpr)
        lw = 2.0 if "QIDF" in name else 1.2
        ls = "-" if "QIDF" in name else "--"
        ax.plot(fpr, tpr, color=d.get("color", "grey"),
                lw=lw, ls=ls, label=f"{name}  (AUC={auc_val:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — FF++ Test Set", pad=8)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    _savefig(fig, os.path.join(out_dir, "fig4_roc_curves.pdf"))


# ---------------------------------------------------------------------------
# Fig 5 — Ablation study
# ---------------------------------------------------------------------------

def fig_ablation(ablation_data: Dict[str, dict], out_dir: str) -> None:
    """
    ablation_data: {"Config A": {"AUC": 0.93, "F1": 0.91}, ...}
    """
    configs = list(ablation_data.keys())
    aucs    = [ablation_data[c]["AUC"] for c in configs]
    f1s     = [ablation_data[c]["F1"]  for c in configs]

    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_auc = ax.bar(x - w / 2, aucs, w, label="AUC",  color="#2166AC", alpha=0.85)
    bars_f1  = ax.bar(x + w / 2, f1s,  w, label="Macro-F1", color="#D6604D", alpha=0.85)

    for bar in [*bars_auc, *bars_f1]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(min(aucs + f1s) - 0.05, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Contribution of Each QIDF-Net Component", pad=8)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    _savefig(fig, os.path.join(out_dir, "fig5_ablation.pdf"))


# ---------------------------------------------------------------------------
# Fig 6 — SHAP summary
# ---------------------------------------------------------------------------

def fig_shap_summary(shap_values, feature_names, out_dir: str) -> None:
    import shap
    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values, feature_names=feature_names,
        plot_type="bar", show=False, color="#2166AC",
        max_display=20,
    )
    plt.title("SHAP Feature Importance — Ensemble Detector", pad=8)
    _savefig(fig, os.path.join(out_dir, "fig6_shap.pdf"))


# ---------------------------------------------------------------------------
# Fig 7 — Attribution confusion matrix
# ---------------------------------------------------------------------------

def fig_attribution_cm(y_true, y_pred, classes, out_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Source")
    ax.set_ylabel("True Source")
    ax.set_title("Attribution Confusion Matrix", pad=8)
    plt.xticks(rotation=25, ha="right")
    _savefig(fig, os.path.join(out_dir, "fig7_attribution_cm.pdf"))


# ---------------------------------------------------------------------------
# Fig 8 — Training convergence (FusionNet)
# ---------------------------------------------------------------------------

def fig_convergence(train_losses: List[float], val_losses: List[float], out_dir: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, color="#2166AC", lw=1.5, label="Training loss")
    ax.plot(epochs, val_losses,   color="#D6604D", lw=1.5, ls="--", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("FusionNet Training Convergence", pad=8)
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "fig8_convergence.pdf"))


# ---------------------------------------------------------------------------
# Fig 9 — Per-manipulation-type F1 comparison
# ---------------------------------------------------------------------------

def fig_per_class_f1(per_class_data: Dict[str, Dict[str, float]], out_dir: str) -> None:
    """
    per_class_data: {"Method A": {"DeepFakes": 0.95, "Face2Face": 0.91, ...}, ...}
    """
    methods = list(per_class_data.keys())
    classes = list(next(iter(per_class_data.values())).keys())
    x = np.arange(len(classes))
    w = 0.8 / len(methods)
    colors = list(PALETTE.values())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, method in enumerate(methods):
        vals = [per_class_data[method].get(c, 0) for c in classes]
        offset = (i - len(methods) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w * 0.9, label=method,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Manipulation-Type F1 Score", pad=8)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    _savefig(fig, os.path.join(out_dir, "fig9_per_class_f1.pdf"))
