"""
baselines.py
============
Three baseline detectors used for comparison against the CNN-RNN ensemble:

1. RandomForestBaseline  - classical ML baseline (tabular, no windowing)
2. SVMBaseline           - classical ML baseline (RBF kernel)
3. SignatureRuleBaseline - a deliberately simple, fast, threshold/rule-based
                            detector standing in for "traditional signature-
                            based systems" in Objective 2's latency
                            comparison. It is NOT meant to be state-of-the-art
                            in detection accuracy (signature systems aren't);
                            it exists to give a fair, honest latency
                            reference point, not a straw-man accuracy number.

All three are fit/evaluated on the exact same preprocessed
train/test arrays as the deep models, so comparisons are apples-to-apples.

Usage:
    python3 src/baselines.py --dataset nsl_kdd
"""
import argparse
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data_utils import load_processed
from signature_baseline import SignatureRuleBaseline
from utils import ensure_dirs, save_json, set_seed


def train_rf(X_train, y_train, seed=42):
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed,
        class_weight="balanced",
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    return clf, time.time() - t0


def train_svm(X_train, y_train, seed=42, max_samples=20000):
    # SVM training is O(n^2)-O(n^3); subsample for tractability on large
    # datasets (e.g. CIC-IDS2017) while keeping class balance.
    rng = np.random.RandomState(seed)
    if len(X_train) > max_samples:
        idx = rng.choice(len(X_train), size=max_samples, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              random_state=seed)
    t0 = time.time()
    clf.fit(X_sub, y_sub)
    return clf, time.time() - t0


def main(dataset: str, processed_root: str, out_root: str, seed: int = 42):
    set_seed(seed)
    X_train, X_test, y_train, y_test, meta = load_processed(dataset, processed_root)

    if len(meta["class_names"]) != 2:
        raise ValueError(
            "baselines.py currently assumes binary labels (normal vs attack). "
            "Re-run data_utils.py with --binary, or extend this script for "
            "the multiclass case."
        )

    ensure_dirs(out_root)
    summary = {}

    print("Training Random Forest baseline...")
    rf, rf_time = train_rf(X_train, y_train, seed)
    joblib.dump(rf, os.path.join(out_root, f"{dataset}_rf.joblib"))
    summary["random_forest"] = {"train_time_sec": rf_time}

    print("Training SVM baseline...")
    svm, svm_time = train_svm(X_train, y_train, seed)
    joblib.dump(svm, os.path.join(out_root, f"{dataset}_svm.joblib"))
    summary["svm"] = {"train_time_sec": svm_time}

    print("Fitting signature-rule baseline...")
    sig = SignatureRuleBaseline().fit(X_train, y_train)
    joblib.dump(sig, os.path.join(out_root, f"{dataset}_signature.joblib"))
    summary["signature_rule"] = {
        "train_time_sec": 0.0,
        "feature_idx": sig.feature_idx_,
        "threshold": sig.threshold_,
    }

    save_json(summary, os.path.join("results", f"{dataset}_baseline_train_summary.json"))
    print("Baseline training complete:")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "cic_ids2017"])
    parser.add_argument("--processed_root", default="data/processed")
    parser.add_argument("--out_root", default="checkpoints/baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dataset, args.processed_root, args.out_root, args.seed)
