"""
run_centralized.py
--------------------
Condition A: Centralized fine-tuning, no privacy mechanism. All NUM_CLIENTS
shards are pooled as if there were no organizational boundary at all; this
is the upper-bound utility reference that DP-FedAvg and CipherTune are
compared against, and it also supplies the "all data was in-training"
member pool for that condition's own membership-inference evaluation.

Usage:
    python -m ciphertune.run_centralized
Writes:
    results/centralized.json
"""

import json
import time
from pathlib import Path

import torch

from . import config as C
from . import partition
from . import model_utils as mu


def run():
    C.set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[centralized] device={device}")

    train_texts, train_labels = partition.load_all_train_pooled()
    test_texts, test_labels = partition.load_test()
    print(f"[centralized] train={len(train_texts)} test={len(test_texts)}")

    model, tokenizer = mu.build_model(device=device)
    n_params = mu.num_trainable_params(model)
    print(f"[centralized] trainable parameters: {n_params}")

    train_dl = mu.make_dataloader(train_texts, train_labels, tokenizer, C.BATCH_SIZE, shuffle=True)
    test_dl = mu.make_dataloader(test_texts, test_labels, tokenizer, C.BATCH_SIZE, shuffle=False)

    t0 = time.perf_counter()
    final_train_loss = None
    for epoch in range(C.CENTRALIZED_EPOCHS):
        final_train_loss = mu.train_local(model, train_dl, epochs=1, lr=C.LEARNING_RATE, device=device)
        print(f"[centralized] epoch {epoch + 1}/{C.CENTRALIZED_EPOCHS} train_loss={final_train_loss:.4f}")
    total_train_time = time.perf_counter() - t0

    acc, f1, test_losses, preds, labels_arr = mu.evaluate(model, test_dl, device=device)
    print(f"[centralized] test accuracy={acc:.4f} macro_f1={f1:.4f}")

    # member losses, for the MIA evaluation: a random subsample of the pooled training set
    member_dl = mu.make_dataloader(
        train_texts[:C.MIA_SAMPLE_SIZE], train_labels[:C.MIA_SAMPLE_SIZE], tokenizer,
        C.BATCH_SIZE, shuffle=False,
    )
    _, _, member_losses, _, _ = mu.evaluate(model, member_dl, device=device)

    Path("results").mkdir(exist_ok=True)
    out = {
        "condition": "centralized",
        "n_trainable_params": n_params,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "final_train_loss": final_train_loss,
        "total_train_time_sec": total_train_time,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "test_losses": test_losses.tolist(),
        "member_losses_for_mia": member_losses.tolist(),
        "nonmember_losses_for_mia": test_losses[:C.MIA_SAMPLE_SIZE].tolist(),
    }
    with open("results/centralized.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[centralized] wrote results/centralized.json")
    return out


if __name__ == "__main__":
    run()
