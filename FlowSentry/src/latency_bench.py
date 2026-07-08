"""
latency_bench.py
================
Measures real per-sample inference latency for every trained model, at
several batch sizes (default: 1, 32, 128 — batch=1 simulates a single-flow,
"edge"-style scoring scenario). Reports mean/median/p95/p99 latency in
milliseconds, on whatever device is available (CPU and/or CUDA).

This produces honest numbers from your actual hardware — it does not
assume or hardcode any particular speedup. If you want a CPU-only "edge
device" comparison, run with CUDA disabled
(CUDA_VISIBLE_DEVICES="" python3 src/latency_bench.py ...).

Usage:
    python3 src/latency_bench.py --dataset nsl_kdd --config configs/config.yaml
"""
import argparse
import os
import time

import joblib
import numpy as np
import torch

from data_utils import load_processed
from datasets import WindowedFlowDataset
from models import build_model
from signature_baseline import SignatureRuleBaseline  # noqa: F401 (needed for joblib.load)
from utils import load_config, get_device, save_json, ensure_dirs


def _percentile(values, p):
    return float(np.percentile(values, p))


def bench_torch_model(model, test_ds, device, batch_size, n_warmup, n_runs):
    model.eval()
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    batches = []
    it = iter(loader)
    for _ in range(min(n_warmup + n_runs, len(loader))):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))

    timings = []
    with torch.no_grad():
        for i, (window, current, _label) in enumerate(batches):
            window, current = window.to(device), current.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(window, current)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= n_warmup:
                timings.append((t1 - t0) * 1000.0 / batch_size)  # ms per sample
    return timings


def bench_sklearn_model(clf, X_test, batch_size, n_warmup, n_runs, predict_fn=None):
    predict_fn = predict_fn or clf.predict
    n = len(X_test)
    timings = []
    for i in range(n_warmup + n_runs):
        start_idx = (i * batch_size) % max(n - batch_size, 1)
        batch = X_test[start_idx:start_idx + batch_size]
        if len(batch) == 0:
            batch = X_test[:batch_size]
        t0 = time.perf_counter()
        _ = predict_fn(batch)
        t1 = time.perf_counter()
        if i >= n_warmup:
            timings.append((t1 - t0) * 1000.0 / max(len(batch), 1))
    return timings


def summarize(timings):
    arr = np.array(timings)
    return {
        "n_measurements": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": _percentile(arr, 95),
        "p99_ms": _percentile(arr, 99),
        "std_ms": float(arr.std()),
    }


def main(dataset: str, config: dict):
    processed_root = config["paths"]["processed_dir"]
    X_train, X_test, y_train, y_test, meta = load_processed(dataset, processed_root)
    n_features = meta["n_features"]
    n_classes = len(meta["class_names"])
    device = get_device(config["latency"]["device_priority"])
    print(f"Benchmarking on device: {device}")

    n_warmup = config["latency"]["n_warmup"]
    n_runs = config["latency"]["n_runs"]
    batch_sizes = config["latency"]["batch_sizes"]

    all_results = {}

    test_ds = WindowedFlowDataset(X_test, y_test, config["data"]["window_length"])
    ckpt_dir = os.path.join(config["paths"]["checkpoints_dir"], dataset)
    for model_name in ["ensemble", "cnn_only", "rnn_only"]:
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
        if not os.path.exists(ckpt_path):
            continue
        model = build_model(model_name, n_features, n_classes, config["model"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        all_results[model_name] = {}
        for bs in batch_sizes:
            timings = bench_torch_model(model, test_ds, device, bs, n_warmup, n_runs)
            all_results[model_name][f"batch_{bs}"] = summarize(timings)
            print(f"{model_name} bs={bs}: mean={all_results[model_name][f'batch_{bs}']['mean_ms']:.4f} ms/sample")

    baseline_dir = "checkpoints/baselines"
    for name, fname in [("random_forest", "rf"), ("svm", "svm"),
                         ("signature_rule", "signature")]:
        path = os.path.join(baseline_dir, f"{dataset}_{fname}.joblib")
        if not os.path.exists(path):
            continue
        clf = joblib.load(path)
        all_results[name] = {}
        for bs in batch_sizes:
            timings = bench_sklearn_model(clf, X_test, bs, n_warmup, n_runs)
            all_results[name][f"batch_{bs}"] = summarize(timings)
            print(f"{name} bs={bs}: mean={all_results[name][f'batch_{bs}']['mean_ms']:.4f} ms/sample")

    all_results["_meta"] = {"device": str(device), "n_warmup": n_warmup, "n_runs": n_runs,
                             "batch_sizes": batch_sizes}

    results_dir = config["paths"]["results_dir"]
    ensure_dirs(results_dir)
    save_json(all_results, os.path.join(results_dir, f"{dataset}_latency.json"))
    print(f"Saved latency benchmark -> {results_dir}/{dataset}_latency.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "cic_ids2017"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(args.dataset, cfg)
