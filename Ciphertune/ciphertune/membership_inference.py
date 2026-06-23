"""
membership_inference.py
-------------------------
Implements the loss-threshold membership-inference attack of Yeom,
Fredrikson, Jha et al., "Privacy Risk in Machine Learning: Analyzing
the Connection to Overfitting" (IEEE CSF, 2018): an attacker who can
query a trained model's per-sample loss on a candidate record guesses
"member" if the loss falls below some threshold and "non-member"
otherwise, since a model's loss on data it was trained on is typically
lower than on unseen data of the same distribution. Sablayrolles et al.,
"White-box vs Black-box: Bayes Optimal Strategies for Membership
Inference" (ICML, 2019) show that, under fairly general assumptions,
exactly this loss-threshold rule is the *Bayes-optimal* membership-
inference strategy -- which is the methodological justification for using
it here instead of a far more expensive shadow-model attack (Shokri et
al., IEEE S&P, 2017): the simple attack is not just cheaper, it is
provably (near-)optimal under the stated assumptions, so there is no
accuracy left on the table by skipping the costlier shadow-model variant.

Rather than a single threshold, we report the full ROC-AUC of
discriminating member vs non-member losses, which is threshold-free and
is the standard metric in this literature:
    AUC = 0.5  -> attacker does no better than a coin flip (no leakage)
    AUC = 1.0  -> attacker perfectly distinguishes members (total leakage)

This module operates purely on the per-sample loss arrays already saved
by each run_*.py script (member_losses_for_mia / nonmember_losses_for_mia)
-- it does not retrain anything, so it can be re-run cheaply on its own.

Usage:
    python -m ciphertune.membership_inference
Writes:
    results/mia_summary.json
"""

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def loss_threshold_mia_auc(member_losses, nonmember_losses) -> float:
    """
    Higher attack score = "model thinks this looks like a training example",
    i.e. LOWER loss => higher membership score, hence we score with -loss.
    """
    member_losses = np.asarray(member_losses)
    nonmember_losses = np.asarray(nonmember_losses)
    scores = np.concatenate([-member_losses, -nonmember_losses])
    true_labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(nonmember_losses)])
    return float(roc_auc_score(true_labels, scores))


def summarize_all(results_dir: str = "results") -> dict:
    summary = {}
    for path in sorted(glob.glob(f"{results_dir}/*.json")):
        if Path(path).name == "mia_summary.json" or Path(path).name == "results.json":
            continue
        with open(path) as f:
            data = json.load(f)
        if "member_losses_for_mia" not in data or "nonmember_losses_for_mia" not in data:
            continue
        auc = loss_threshold_mia_auc(data["member_losses_for_mia"], data["nonmember_losses_for_mia"])
        key = Path(path).stem
        summary[key] = {
            "condition": data.get("condition"),
            "epsilon_round": data.get("epsilon_round"),
            "mia_auc": auc,
            "n_members": len(data["member_losses_for_mia"]),
            "n_nonmembers": len(data["nonmember_losses_for_mia"]),
            "mean_member_loss": float(np.mean(data["member_losses_for_mia"])),
            "mean_nonmember_loss": float(np.mean(data["nonmember_losses_for_mia"])),
        }
        print(f"[mia] {key}: AUC={auc:.4f}  "
              f"mean_member_loss={summary[key]['mean_member_loss']:.4f}  "
              f"mean_nonmember_loss={summary[key]['mean_nonmember_loss']:.4f}")

    with open(f"{results_dir}/mia_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[mia] wrote {results_dir}/mia_summary.json")
    return summary


if __name__ == "__main__":
    summarize_all()
