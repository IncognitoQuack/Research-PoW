"""
make_figures.py
===============
Generates every figure from the JSON/CSV files written into results/ by
train.py / evaluate.py / latency_bench.py. Never invents numbers — if a
results file is missing, that figure is skipped with a warning instead of
being drawn with placeholder data.

Usage:
    python3 src/make_figures.py --dataset nsl_kdd
    python3 src/make_figures.py --dataset nsl_kdd --figures_dir figures
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

MODEL_LABELS = {
    "ensemble": "CNN-RNN Ensemble (proposed)",
    "cnn_only": "CNN-only",
    "rnn_only": "RNN-only",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "signature_rule": "Signature-based (rule)",
}


def _load_json(path):
    if not os.path.exists(path):
        print(f"[skip] missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_curves(results_dir, figures_dir, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plotted = False
    for model_name in ["ensemble", "cnn_only", "rnn_only"]:
        hist = _load_json(os.path.join(results_dir, f"{dataset}_{model_name}_history.json"))
        if hist is None:
            continue
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], "--", alpha=0.7,
                     label=f"{MODEL_LABELS[model_name]} (train)")
        axes[0].plot(epochs, hist["val_loss"], "-", label=f"{MODEL_LABELS[model_name]} (val)")
        axes[1].plot(epochs, hist["train_acc"], "--", alpha=0.7,
                     label=f"{MODEL_LABELS[model_name]} (train)")
        axes[1].plot(epochs, hist["val_acc"], "-", label=f"{MODEL_LABELS[model_name]} (val)")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training / Validation Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Training / Validation Accuracy")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle(f"Training Convergence — {dataset}")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{dataset}_training_curves.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_metrics_bar(results_dir, figures_dir, dataset):
    metrics = _load_json(os.path.join(results_dir, f"{dataset}_metrics.json"))
    if metrics is None:
        return
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "model": MODEL_LABELS.get(model_name, model_name),
            "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"],
            "FPR": m["false_positive_rate"],
        })
    df = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    df[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Detection Performance Comparison — {dataset}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{dataset}_metrics_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    fig, ax = plt.subplots(figsize=(9, 5))
    df["FPR"].plot(kind="bar", ax=ax, color="firebrick")
    ax.set_ylabel("False Positive Rate")
    ax.set_title(f"False Positive Rate Comparison — {dataset}")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{dataset}_fpr_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_confusion_matrices(results_dir, figures_dir, dataset):
    metrics = _load_json(os.path.join(results_dir, f"{dataset}_metrics.json"))
    if metrics is None:
        return
    models = list(metrics.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        cm = np.array(metrics[model_name]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle(f"Confusion Matrices — {dataset}")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{dataset}_confusion_matrices.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_latency_comparison(results_dir, figures_dir, dataset):
    lat = _load_json(os.path.join(results_dir, f"{dataset}_latency.json"))
    if lat is None:
        return
    batch_sizes = lat.get("_meta", {}).get("batch_sizes", [1])
    rows = []
    for model_name, per_bs in lat.items():
        if model_name == "_meta":
            continue
        for bs in batch_sizes:
            key = f"batch_{bs}"
            if key in per_bs:
                rows.append({
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "batch_size": bs,
                    "mean_ms": per_bs[key]["mean_ms"],
                    "p95_ms": per_bs[key]["p95_ms"],
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df, x="model", y="mean_ms", hue="batch_size", ax=ax)
    ax.set_ylabel("Mean inference latency (ms/sample)")
    ax.set_title(f"Inference Latency Comparison — {dataset} (device: "
                 f"{lat['_meta'].get('device', 'unknown')})")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{dataset}_latency_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_ablation(results_dir, figures_dir, dataset):
    """Ablation: ensemble vs. its two single-branch components, accuracy + FPR."""
    metrics = _load_json(os.path.join(results_dir, f"{dataset}_metrics.json"))
    if metrics is None:
        return
    keys = [k for k in ["ensemble", "cnn_only", "rnn_only"] if k in metrics]
    if len(keys) < 2:
        return
    rows = [{"model": MODEL_LABELS[k], "Accuracy": metrics[k]["accuracy"],
             "F1": metrics[k]["f1"], "FPR": metrics[k]["false_positive_rate"]}
            for k in keys]
    df = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_title(f"Ablation: Fusion vs. Single-Branch Models — {dataset}")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{dataset}_ablation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main(dataset: str, results_dir: str, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(results_dir, figures_dir, dataset)
    plot_metrics_bar(results_dir, figures_dir, dataset)
    plot_confusion_matrices(results_dir, figures_dir, dataset)
    plot_latency_comparison(results_dir, figures_dir, dataset)
    plot_ablation(results_dir, figures_dir, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "cic_ids2017"])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()
    main(args.dataset, args.results_dir, args.figures_dir)
