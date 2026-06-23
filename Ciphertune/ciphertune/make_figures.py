"""
make_figures.py
-----------------
Generates the figures used in the paper, reading ONLY from
results/results.json -- the consolidated output of run_all.py after it
has been run on REAL data on your machine. This script intentionally
refuses to run if that file does not exist, so there is no risk of ever
accidentally generating a "placeholder" figure that could be mistaken
for a real result.

Figures produced (saved to figures/):
  fig_utility.png        -- grouped bar chart: accuracy & macro-F1 across
                             Centralized / DP-FedAvg (sweep) / CipherTune
  fig_privacy_mia.png    -- bar chart: membership-inference AUC across
                             the same conditions, with a reference line
                             at 0.5 (no leakage)
  fig_overhead_time.png  -- bar chart: total federated training wall-clock
                             time across conditions
  fig_overhead_bytes.png -- bar chart: bytes transmitted per round,
                             DP-FedAvg (plaintext) vs CipherTune (ciphertext)

Usage:
    python -m ciphertune.make_figures
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import config as C

RESULTS_PATH = "results/results.json"
FIG_DIR = "figures"


def _load_results():
    if not Path(RESULTS_PATH).exists():
        raise SystemExit(
            f"[make_figures] {RESULTS_PATH} not found. Run `python -m ciphertune.run_all` "
            f"on real data first -- this script will not fabricate placeholder figures."
        )
    with open(RESULTS_PATH) as f:
        return json.load(f)


def fig_utility(results: dict):
    conditions = ["Centralized"]
    accs = [results["centralized"]["test_accuracy"]]
    f1s = [results["centralized"]["test_macro_f1"]]

    for eps_round, d in results["dp_fedavg"].items():
        conditions.append(f"DP-FedAvg\n(eps={eps_round})")
        accs.append(d["test_accuracy"])
        f1s.append(d["test_macro_f1"])

    conditions.append("CipherTune\n(ours)")
    accs.append(results["ciphertune"]["test_accuracy"])
    f1s.append(results["ciphertune"]["test_macro_f1"])

    x = range(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], accs, width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], f1s, width, label="Macro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Utility: CVE severity classification accuracy and macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_utility.png", dpi=300)
    plt.close(fig)


def fig_privacy_mia(results: dict):
    mia = results["mia_summary"]
    labels, aucs = [], []
    for key, val in mia.items():
        if val["condition"] == "centralized":
            labels.append("Centralized")
        elif val["condition"] == "dp_fedavg":
            labels.append(f"DP-FedAvg\n(eps={val['epsilon_round']})")
        elif val["condition"] == "ciphertune":
            labels.append("CipherTune\n(ours)")
        else:
            labels.append(key)
        aucs.append(val["mia_auc"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, aucs, color="indianred")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="No leakage (AUC=0.5)")
    ax.set_ylabel("Membership-inference attack AUC")
    ax.set_ylim(0, 1.0)
    ax.set_title("Privacy: loss-threshold membership-inference attack AUC")
    ax.legend()
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_privacy_mia.png", dpi=300)
    plt.close(fig)


def fig_overhead_time(results: dict):
    labels = ["Centralized"]
    times = [results["centralized"]["total_train_time_sec"]]
    for eps_round, d in results["dp_fedavg"].items():
        labels.append(f"DP-FedAvg\n(eps={eps_round})")
        times.append(d["total_train_time_sec"])
    labels.append("CipherTune\n(ours)")
    times.append(results["ciphertune"]["total_train_time_sec"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, times, color="steelblue")
    ax.set_ylabel("Total training wall-clock time (seconds)")
    ax.set_title(f"Overhead: end-to-end training time "
                 f"({C.FEDERATED_ROUNDS} federated rounds / {C.CENTRALIZED_EPOCHS} centralized epochs)")
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_overhead_time.png", dpi=300)
    plt.close(fig)


def fig_overhead_bytes(results: dict):
    labels, byte_values = [], []
    for eps_round, d in results["dp_fedavg"].items():
        labels.append(f"DP-FedAvg\n(eps={eps_round})\n[plaintext]")
        byte_values.append(d["bytes_transmitted_per_round"][0])
    labels.append("CipherTune\n(ours)\n[ciphertext]")
    byte_values.append(results["ciphertune"]["bytes_transmitted_per_round"][0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [b / 1024 for b in byte_values], color="seagreen")
    ax.set_ylabel("Bytes transmitted per round (KiB)")
    ax.set_title("Communication overhead per federated round (all clients combined)\n"
                  "Centralized has no network transmission and is omitted")
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_overhead_bytes.png", dpi=300)
    plt.close(fig)


def main():
    Path(FIG_DIR).mkdir(exist_ok=True)
    results = _load_results()
    fig_utility(results)
    fig_privacy_mia(results)
    fig_overhead_time(results)
    fig_overhead_bytes(results)
    print(f"[make_figures] wrote 4 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
