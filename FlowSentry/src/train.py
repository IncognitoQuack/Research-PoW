"""
train.py
========
Trains the CNN-RNN ensemble and the two single-branch ablation models
(CNN-only, RNN-only) on a preprocessed dataset, with early stopping on
validation loss. Saves checkpoints to `checkpoints/<dataset>/<model>.pt`
and training curves to `results/<dataset>_<model>_history.json`.

Usage:
    python3 src/train.py --dataset nsl_kdd --config configs/config.yaml
    python3 src/train.py --dataset nsl_kdd --config configs/config.yaml --model cnn_only
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data_utils import load_processed
from datasets import make_loaders
from models import build_model
from utils import set_seed, load_config, get_device, save_json, ensure_dirs


def class_weights_from_labels(y_train: np.ndarray, n_classes: int, device):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, desc: str = ""):
    model.train(mode=train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    iterator = tqdm(loader, desc=desc, leave=False, mininterval=1.0) if train else loader
    with ctx:
        for window, current, label in iterator:
            window, current, label = window.to(device), current.to(device), label.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(window, current)
            loss = criterion(logits, label)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * label.size(0)
            total_correct += (logits.argmax(dim=-1) == label).sum().item()
            total_n += label.size(0)
            if train:
                iterator.set_postfix(loss=f"{total_loss / total_n:.4f}",
                                      acc=f"{total_correct / total_n:.4f}")
    return total_loss / total_n, total_correct / total_n


def train_model(dataset: str, model_name: str, config: dict, seed: int = None):
    seed = seed if seed is not None else config.get("seed", 42)
    set_seed(seed)
    device = get_device(config["latency"]["device_priority"])

    processed_root = config["paths"]["processed_dir"]
    X_train, X_test, y_train, y_test, meta = load_processed(dataset, processed_root)
    n_features = meta["n_features"]
    n_classes = len(meta["class_names"])

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test,
        window_length=config["data"]["window_length"],
        batch_size=config["train"]["batch_size"],
        val_fraction=config["data"]["val_size"],
        seed=seed,
        num_workers=config["train"].get("num_workers", 0),
    )

    model = build_model(model_name, n_features, n_classes, config["model"]).to(device)
    weights = class_weights_from_labels(y_train, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"],
                      weight_decay=config["train"]["weight_decay"])

    print(f"[{dataset}/{model_name}] device={device} | train_samples={len(train_loader.dataset)} "
          f"| batch_size={config['train']['batch_size']} | steps/epoch={len(train_loader)} "
          f"| num_workers={config['train'].get('num_workers', 0)}")

    best_val_loss = float("inf")
    patience = config["train"]["early_stopping_patience"]
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    ckpt_dir = os.path.join(config["paths"]["checkpoints_dir"], dataset)
    ensure_dirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")

    start = time.time()
    for epoch in range(1, config["train"]["epochs"] + 1):
        epoch_t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                     train=True, desc=f"{model_name} epoch {epoch}")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        epoch_time = time.time() - epoch_t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{dataset}/{model_name}] epoch {epoch:03d} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({epoch_time:.1f}s/epoch)")
        if epoch == 1:
            remaining = epoch_time * (config["train"]["epochs"] - 1)
            print(f"[{dataset}/{model_name}] estimated remaining time if no early stop: "
                  f"~{remaining / 60:.1f} min ({config['train']['epochs'] - 1} epochs left)")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    train_time = time.time() - start
    history["train_time_sec"] = train_time
    history["best_val_loss"] = best_val_loss
    history["n_epochs_run"] = len(history["train_loss"])

    results_dir = config["paths"]["results_dir"]
    ensure_dirs(results_dir)
    save_json(history, os.path.join(results_dir, f"{dataset}_{model_name}_history.json"))

    print(f"Done. Best checkpoint: {ckpt_path} (val_loss={best_val_loss:.4f}, "
          f"train_time={train_time:.1f}s)")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "cic_ids2017"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="ensemble",
                         choices=["ensemble", "cnn_only", "rnn_only"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_model(args.dataset, args.model, cfg, seed=args.seed)
