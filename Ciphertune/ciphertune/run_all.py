"""
run_all.py
-----------
Single entry point that runs the full CipherTune experimental pipeline
end to end, in the correct order, and consolidates every condition's
output into one results/results.json plus a flat results/summary_table.csv
that maps directly onto the paper's main results table.

This script assumes:
  1. You have already run `python -m ciphertune.data_fetch` once
     (needs your own internet access to the real NVD API).
  2. You have already run `python -m ciphertune.partition` once
     (or this script will call it automatically if splits are missing).

Usage:
    python -m ciphertune.run_all
"""

import json
from pathlib import Path

import pandas as pd

from . import config as C
from . import partition
from . import run_centralized
from . import run_dp_fedavg
from . import run_ciphertune
from . import membership_inference as mia


def ensure_splits():
    if not Path(C.SPLITS_DIR, "test.csv").exists():
        print("[run_all] splits not found -- running partition.make_splits() now.")
        partition.make_splits()
    else:
        print("[run_all] found existing splits in", C.SPLITS_DIR)


def main():
    if not Path(C.PROCESSED_CSV).exists():
        raise SystemExit(
            f"[run_all] {C.PROCESSED_CSV} not found. Run `python -m ciphertune.data_fetch` "
            f"first (requires your own internet access to the real NVD API)."
        )

    ensure_splits()

    print("\n========== CONDITION A: CENTRALIZED ==========")
    centralized_out = run_centralized.run()

    print("\n========== CONDITION B: DP-FEDAVG (epsilon sweep) ==========")
    dp_out = run_dp_fedavg.run()

    print("\n========== CONDITION C: CIPHERTUNE (ours) ==========")
    ciphertune_out = run_ciphertune.run()

    print("\n========== MEMBERSHIP INFERENCE ATTACK ==========")
    mia_out = mia.summarize_all()

    consolidated = {
        "config_snapshot": {
            "num_clients": C.NUM_CLIENTS,
            "federated_rounds": C.FEDERATED_ROUNDS,
            "local_epochs_per_round": C.LOCAL_EPOCHS_PER_ROUND,
            "lora_r": C.LORA_R,
            "lora_target_layers": C.LORA_TARGET_LAYER_INDICES,
            "paillier_key_bits": C.PAILLIER_KEY_BITS,
            "quant_bits": C.QUANT_BITS,
            "dp_epsilon_round_sweep": C.DP_EPSILON_ROUND_SWEEP,
            "dp_headline_epsilon_round": C.DP_HEADLINE_EPSILON_ROUND,
        },
        "centralized": centralized_out,
        "dp_fedavg": dp_out,
        "ciphertune": ciphertune_out,
        "mia_summary": mia_out,
    }
    with open("results/results.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    print("\n[run_all] wrote results/results.json")

    build_summary_table(consolidated)


def build_summary_table(consolidated: dict):
    rows = []

    c = consolidated["centralized"]
    rows.append({
        "condition": "Centralized",
        "epsilon_total": None,
        "test_accuracy": c["test_accuracy"],
        "test_macro_f1": c["test_macro_f1"],
        "mia_auc": consolidated["mia_summary"].get("centralized", {}).get("mia_auc"),
        "total_train_time_sec": c["total_train_time_sec"],
        "bytes_per_round": None,
    })

    headline_key = f"dp_fedavg_eps{C.DP_HEADLINE_EPSILON_ROUND}"
    for eps_round, d in consolidated["dp_fedavg"].items():
        rows.append({
            "condition": f"DP-FedAvg (eps_round={eps_round})",
            "epsilon_total": d["epsilon_total_basic_composition"],
            "test_accuracy": d["test_accuracy"],
            "test_macro_f1": d["test_macro_f1"],
            "mia_auc": consolidated["mia_summary"].get(f"dp_fedavg_eps{eps_round}", {}).get("mia_auc"),
            "total_train_time_sec": d["total_train_time_sec"],
            "bytes_per_round": d["bytes_transmitted_per_round"][0] if d["bytes_transmitted_per_round"] else None,
        })

    ct = consolidated["ciphertune"]
    rows.append({
        "condition": "CipherTune (ours)",
        "epsilon_total": None,
        "test_accuracy": ct["test_accuracy"],
        "test_macro_f1": ct["test_macro_f1"],
        "mia_auc": consolidated["mia_summary"].get("ciphertune", {}).get("mia_auc"),
        "total_train_time_sec": ct["total_train_time_sec"],
        "bytes_per_round": ct["bytes_transmitted_per_round"][0] if ct["bytes_transmitted_per_round"] else None,
    })

    df = pd.DataFrame(rows)
    df.to_csv("results/summary_table.csv", index=False)
    print("[run_all] wrote results/summary_table.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
