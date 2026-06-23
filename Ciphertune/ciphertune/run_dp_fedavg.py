"""
run_dp_fedavg.py
------------------
Condition B: DP-FedAvg baseline. Federated averaging of LoRA deltas with
client-level differential privacy (clip + single server-side Gaussian
noise addition to the averaged update; see dp_utils.py for the full
construction and the rationale for reporting an explicit epsilon sweep
rather than one fixed operating point).

No encryption is used in this baseline -- clients send their (locally
trained, plaintext) LoRA delta to the server every round, and the server
clips/noises/averages it. This isolates the *DP* contribution to privacy
from the *encryption* contribution, which is exactly the comparison
Objective 2 calls for.

Usage:
    python -m ciphertune.run_dp_fedavg
Writes:
    results/dp_fedavg_eps{epsilon_round}.json   (one file per sweep point)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import partition
from . import model_utils as mu
from . import dp_utils as dp


def run_one_epsilon(epsilon_round: float, device: str):
    print(f"\n[dp_fedavg] ==== epsilon_round={epsilon_round} ====")
    rng = np.random.default_rng(C.SEED)

    global_model, tokenizer = mu.build_model(device=device)
    global_vec, shapes = mu.get_trainable_vector(global_model)
    n_params = len(global_vec)
    print(f"[dp_fedavg] trainable parameters: {n_params}")

    test_texts, test_labels = partition.load_test()
    test_dl = mu.make_dataloader(test_texts, test_labels, tokenizer, C.BATCH_SIZE, shuffle=False)

    client_data = [partition.load_client(i) for i in range(C.NUM_CLIENTS)]

    round_times = []
    round_sigmas = []
    bytes_per_round = []
    plaintext_vector_bytes = n_params * 4  # float32

    t_total0 = time.perf_counter()
    for rnd in range(C.FEDERATED_ROUNDS):
        t_round0 = time.perf_counter()
        client_deltas = []
        for texts, labels in client_data:
            local_model, _ = mu.build_model(device=device)
            mu.set_trainable_vector(local_model, global_vec, shapes)
            dl = mu.make_dataloader(texts, labels, tokenizer, C.BATCH_SIZE, shuffle=True)
            mu.train_local(local_model, dl, epochs=C.LOCAL_EPOCHS_PER_ROUND, lr=C.LEARNING_RATE, device=device)
            local_vec, _ = mu.get_trainable_vector(local_model)
            client_deltas.append(local_vec - global_vec)

        noised_avg_delta, sigma = dp.aggregate_with_central_dp(client_deltas, epsilon_round, rng)
        global_vec = global_vec + noised_avg_delta
        mu.set_trainable_vector(global_model, global_vec, shapes)

        round_time = time.perf_counter() - t_round0
        round_times.append(round_time)
        round_sigmas.append(sigma)
        # plaintext baseline: every client transmits its full plaintext float32 delta
        bytes_per_round.append(plaintext_vector_bytes * C.NUM_CLIENTS)
        print(f"[dp_fedavg] round {rnd + 1}/{C.FEDERATED_ROUNDS} "
              f"sigma={sigma:.5f} round_time={round_time:.2f}s")

    total_time = time.perf_counter() - t_total0

    acc, f1, test_losses, preds, labels_arr = mu.evaluate(global_model, test_dl, device=device)
    print(f"[dp_fedavg] eps_round={epsilon_round} -> test accuracy={acc:.4f} macro_f1={f1:.4f}")

    member_texts, member_labels = [], []
    per_client_take = max(1, C.MIA_SAMPLE_SIZE // C.NUM_CLIENTS)
    for texts, labels in client_data:
        member_texts.extend(texts[:per_client_take])
        member_labels.extend(labels[:per_client_take])
    member_dl = mu.make_dataloader(member_texts, member_labels, tokenizer, C.BATCH_SIZE, shuffle=False)
    _, _, member_losses, _, _ = mu.evaluate(global_model, member_dl, device=device)

    eps_total = dp.total_epsilon_basic(epsilon_round, C.FEDERATED_ROUNDS)
    delta_total = dp.total_delta_basic(C.DP_DELTA_PER_ROUND, C.FEDERATED_ROUNDS)

    out = {
        "condition": "dp_fedavg",
        "epsilon_round": epsilon_round,
        "epsilon_total_basic_composition": eps_total,
        "delta_total_basic_composition": delta_total,
        "n_trainable_params": n_params,
        "num_clients": C.NUM_CLIENTS,
        "federated_rounds": C.FEDERATED_ROUNDS,
        "round_times_sec": round_times,
        "total_train_time_sec": total_time,
        "round_sigmas": round_sigmas,
        "bytes_transmitted_per_round": bytes_per_round,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "test_losses": test_losses.tolist(),
        "member_losses_for_mia": member_losses.tolist(),
        "nonmember_losses_for_mia": test_losses[:len(member_losses)].tolist(),
    }
    Path("results").mkdir(exist_ok=True)
    out_path = f"results/dp_fedavg_eps{epsilon_round}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[dp_fedavg] wrote {out_path}")
    return out


def run():
    C.set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dp_fedavg] device={device}")
    results = {}
    for eps_round in C.DP_EPSILON_ROUND_SWEEP:
        results[eps_round] = run_one_epsilon(eps_round, device)
    return results


if __name__ == "__main__":
    run()
