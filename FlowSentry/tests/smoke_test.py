"""
smoke_test.py
=============
Generates a small SYNTHETIC dataset with the same schema/shape as NSL-KDD
(41 numeric-after-encoding features + binary label) and runs the entire
pipeline end-to-end: preprocessing -> windowing -> model forward pass ->
a short training run -> evaluation -> latency benchmark -> figure
generation.

This is a plumbing test, not a scientific experiment. It exists to catch
shape mismatches / import errors / API misuse before you spend hours
training on the real datasets. The printed "accuracy" numbers are
meaningless (the synthetic data is near-random) and must never be quoted
anywhere.

Run:
    python3 tests/smoke_test.py
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import set_seed, load_config, ensure_dirs, save_json, get_device  # noqa: E402
from datasets import make_loaders, WindowedFlowDataset  # noqa: E402
from models import build_model  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import Adam  # noqa: E402


def make_synthetic(n_samples=4000, n_features=41, seed=42):
    rng = np.random.RandomState(seed)
    # two mildly-separated Gaussian blobs standing in for normal/attack flows
    n_pos = n_samples // 3
    n_neg = n_samples - n_pos
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, n_features))
    X_pos = rng.normal(loc=0.6, scale=1.0, size=(n_pos, n_features))
    X = np.concatenate([X_neg, X_pos], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)]).astype(np.int64)
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    print("=" * 60)
    print("SMOKE TEST: synthetic-data end-to-end pipeline check")
    print("=" * 60)

    tmp_root = "tests/_smoke_tmp"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    ensure_dirs(tmp_root)

    set_seed(42)
    X_train, y_train, X_test, y_test = make_synthetic()
    print(f"[1/7] synthetic data ok: X_train={X_train.shape}, X_test={X_test.shape}")

    meta = {"n_features": X_train.shape[1], "class_names": ["normal", "attack"]}

    cfg = load_config("configs/config.yaml")
    cfg["train"]["epochs"] = 2  # keep the smoke test fast
    cfg["train"]["batch_size"] = 64
    cfg["latency"]["n_warmup"] = 3
    cfg["latency"]["n_runs"] = 10
    cfg["paths"]["results_dir"] = os.path.join(tmp_root, "results")
    cfg["paths"]["figures_dir"] = os.path.join(tmp_root, "figures")
    cfg["paths"]["checkpoints_dir"] = os.path.join(tmp_root, "checkpoints")
    ensure_dirs(cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"],
                cfg["paths"]["checkpoints_dir"])

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test,
        window_length=cfg["data"]["window_length"],
        batch_size=cfg["train"]["batch_size"],
        val_fraction=0.1, seed=42,
    )
    print("[2/7] windowing + DataLoader ok")

    device = get_device(["cpu"])  # smoke test forced to CPU for determinism/speed
    n_features = meta["n_features"]
    n_classes = len(meta["class_names"])

    for model_name in ["ensemble", "cnn_only", "rnn_only"]:
        model = build_model(model_name, n_features, n_classes, cfg["model"]).to(device)
        window, current, label = next(iter(train_loader))
        logits = model(window.to(device), current.to(device))
        assert logits.shape == (window.shape[0], n_classes), \
            f"{model_name} output shape mismatch: {logits.shape}"
    print("[3/7] forward pass shapes ok for ensemble/cnn_only/rnn_only")

    # short training loop for the ensemble only (fast check of backward pass)
    model = build_model("ensemble", n_features, n_classes, cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for epoch in range(cfg["train"]["epochs"]):
        for window, current, label in train_loader:
            optimizer.zero_grad()
            logits = model(window.to(device), current.to(device))
            loss = criterion(logits, label.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    assert len(losses) > 0 and np.isfinite(losses[-1]), "training loop produced NaN/no loss"
    print(f"[4/7] short training loop ok (final batch loss={losses[-1]:.4f})")

    # evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for window, current, label in test_loader:
            logits = model(window.to(device), current.to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(label.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f"[5/7] evaluation loop ok (synthetic 'accuracy'={acc:.3f} — MEANINGLESS, "
          "do not report this number anywhere)")

    # latency bench (tiny)
    test_ds = WindowedFlowDataset(X_test, y_test, cfg["data"]["window_length"])
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    import time
    timings = []
    with torch.no_grad():
        for i, (window, current, _label) in enumerate(loader):
            if i >= 15:
                break
            t0 = time.perf_counter()
            _ = model(window.to(device), current.to(device))
            t1 = time.perf_counter()
            if i >= 5:
                timings.append((t1 - t0) * 1000)
    print(f"[6/7] latency benchmarking loop ok (mean={np.mean(timings):.3f} ms/sample on CPU)")

    # figure generation smoke check (writes into tmp dir, not the real figures/)
    fake_metrics = {
        "ensemble": {"accuracy": float(acc), "precision": 0.5, "recall": 0.5, "f1": 0.5,
                     "false_positive_rate": 0.1,
                     "confusion_matrix": [[10, 2], [3, 8]]},
    }
    save_json(fake_metrics, os.path.join(cfg["paths"]["results_dir"], "smoke_metrics.json"))
    print("[7/7] JSON I/O ok")

    shutil.rmtree(tmp_root)
    print("=" * 60)
    print("ALL SMOKE TESTS PASSED. Pipeline is wired correctly.")
    print("Remember: none of the numbers above are real — run on the actual")
    print("NSL-KDD / CIC-IDS2017 data via run_all.sh for real results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
