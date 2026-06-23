"""
run_ciphertune.py
-------------------
Condition C (ours): CipherTune. Federated LoRA averaging in which every
client's locally-trained delta is quantized, packed, and Paillier-
encrypted BEFORE it ever leaves the client; the aggregating server is
mathematically restricted to (a) summing ciphertexts homomorphically and
(b) decrypting only the resulting cohort-level SUM, never an individual
client's contribution. See crypto_utils.py for the packing/encryption
scheme itself.

No differential-privacy noise is injected here -- encryption is the
privacy mechanism for THIS condition, and it protects a different threat
model than DP does: confidentiality of each update from the aggregator
and from anyone observing the network, not statistical leakage from the
released model itself. We deliberately do not conflate the two; the
Discussion section of the paper makes this distinction explicit, and the
membership-inference results (run separately, see membership_inference.py)
are exactly the experiment that reveals whether this distinction matters
empirically.

Usage:
    python -m ciphertune.run_ciphertune
Writes:
    results/ciphertune.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import partition
from . import model_utils as mu
from . import crypto_utils as cu


def run():
    C.set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ciphertune] device={device}")

    global_model, tokenizer = mu.build_model(device=device)
    global_vec, shapes = mu.get_trainable_vector(global_model)
    n_params = len(global_vec)
    print(f"[ciphertune] trainable parameters: {n_params}")

    test_texts, test_labels = partition.load_test()
    test_dl = mu.make_dataloader(test_texts, test_labels, tokenizer, C.BATCH_SIZE, shuffle=False)

    client_data = [partition.load_client(i) for i in range(C.NUM_CLIENTS)]

    quantizer = cu.Quantizer(bits=C.QUANT_BITS, scale=C.QUANT_SCALE)
    aggregator = cu.PaillierAggregator(key_bits=C.PAILLIER_KEY_BITS)
    packer = cu.Packer(bits=C.QUANT_BITS, num_clients=C.NUM_CLIENTS,
                        margin_bits=C.PACK_MARGIN_BITS, modulus_bits=C.PAILLIER_KEY_BITS)
    print(f"[ciphertune] paillier key_bits={C.PAILLIER_KEY_BITS} "
          f"slot_width={packer.slot_width} slots_per_chunk={packer.slots_per_chunk} "
          f"num_chunks={packer.num_chunks(n_params)}")

    round_times = []
    enc_times_per_round = []
    sum_times_per_round = []
    dec_times_per_round = []
    bytes_per_round = []
    saturation_rates = []

    t_total0 = time.perf_counter()
    for rnd in range(C.FEDERATED_ROUNDS):
        t_round0 = time.perf_counter()
        client_ciphertexts = []
        round_enc_time = 0.0
        round_bytes = 0
        for texts, labels in client_data:
            local_model, _ = mu.build_model(device=device)
            mu.set_trainable_vector(local_model, global_vec, shapes)
            dl = mu.make_dataloader(texts, labels, tokenizer, C.BATCH_SIZE, shuffle=True)
            mu.train_local(local_model, dl, epochs=C.LOCAL_EPOCHS_PER_ROUND, lr=C.LEARNING_RATE, device=device)
            local_vec, _ = mu.get_trainable_vector(local_model)
            delta = local_vec - global_vec

            ciphertexts, enc_time = cu.encrypt_delta(delta, quantizer, packer, aggregator)
            client_ciphertexts.append(ciphertexts)
            round_enc_time += enc_time
            round_bytes += aggregator.serialized_ciphertext_bytes(ciphertexts)

        avg_delta, sum_time, dec_time = cu.aggregate_and_decrypt(
            client_ciphertexts, n_params, C.NUM_CLIENTS, quantizer, packer, aggregator
        )
        global_vec = global_vec + avg_delta
        mu.set_trainable_vector(global_model, global_vec, shapes)

        round_time = time.perf_counter() - t_round0
        round_times.append(round_time)
        enc_times_per_round.append(round_enc_time)
        sum_times_per_round.append(sum_time)
        dec_times_per_round.append(dec_time)
        bytes_per_round.append(round_bytes)
        saturation_rates.append(quantizer.saturation_rate())

        print(f"[ciphertune] round {rnd + 1}/{C.FEDERATED_ROUNDS} "
              f"round_time={round_time:.2f}s enc_time={round_enc_time:.2f}s "
              f"sum_time={sum_time:.4f}s dec_time={dec_time:.2f}s "
              f"bytes={round_bytes} saturation={quantizer.saturation_rate():.4%}")

    total_time = time.perf_counter() - t_total0

    acc, f1, test_losses, preds, labels_arr = mu.evaluate(global_model, test_dl, device=device)
    print(f"[ciphertune] test accuracy={acc:.4f} macro_f1={f1:.4f}")

    member_texts, member_labels = [], []
    per_client_take = max(1, C.MIA_SAMPLE_SIZE // C.NUM_CLIENTS)
    for texts, labels in client_data:
        member_texts.extend(texts[:per_client_take])
        member_labels.extend(labels[:per_client_take])
    member_dl = mu.make_dataloader(member_texts, member_labels, tokenizer, C.BATCH_SIZE, shuffle=False)
    _, _, member_losses, _, _ = mu.evaluate(global_model, member_dl, device=device)

    out = {
        "condition": "ciphertune",
        "n_trainable_params": n_params,
        "num_clients": C.NUM_CLIENTS,
        "federated_rounds": C.FEDERATED_ROUNDS,
        "paillier_key_bits": C.PAILLIER_KEY_BITS,
        "quant_bits": C.QUANT_BITS,
        "quant_scale": C.QUANT_SCALE,
        "slot_width": packer.slot_width,
        "slots_per_chunk": packer.slots_per_chunk,
        "num_chunks": packer.num_chunks(n_params),
        "round_times_sec": round_times,
        "total_train_time_sec": total_time,
        "encryption_time_sec_per_round": enc_times_per_round,
        "homomorphic_sum_time_sec_per_round": sum_times_per_round,
        "decryption_time_sec_per_round": dec_times_per_round,
        "bytes_transmitted_per_round": bytes_per_round,
        "quantization_saturation_rate_per_round": saturation_rates,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "test_losses": test_losses.tolist(),
        "member_losses_for_mia": member_losses.tolist(),
        "nonmember_losses_for_mia": test_losses[:len(member_losses)].tolist(),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/ciphertune.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[ciphertune] wrote results/ciphertune.json")
    return out


if __name__ == "__main__":
    run()
