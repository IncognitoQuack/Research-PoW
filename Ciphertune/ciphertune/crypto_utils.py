"""
crypto_utils.py
----------------
Implements the quantization-aware Paillier secure-aggregation scheme used
by CipherTune (Objective 1 / Contribution 2).

Why packing is necessary
-------------------------
Paillier is an additively homomorphic scheme: given ciphertexts c1 = Enc(m1)
and c2 = Enc(m2) under the same public key, c1 * c2 mod n^2 decrypts to
m1 + m2. This lets a server sum encrypted client updates without ever
seeing an individual client's plaintext update. However, encrypting every
single LoRA-delta scalar as its own ciphertext is far too slow for a
multi-thousand-parameter update (each Paillier operation costs several
milliseconds because the modulus is hundreds of bits wide). Because the
Paillier plaintext space is enormous (an n-bit integer, n=1024 or 2048),
we instead pack several quantized values into ONE plaintext integer using
a fixed-width base-2^k positional encoding, analogous to the ciphertext
packing strategy used in additively-homomorphic secure-aggregation systems
for distributed learning (see Aono et al., 2018, IEEE TIFS, DOI listed in
the manuscript references). Because addition of packed integers respects
per-slot addition as long as no slot overflows into its neighbour, summing
the *packed* plaintexts via the homomorphic property simultaneously sums
every one of the underlying quantized parameters with a single Paillier
ciphertext addition.

This module is deliberately self-contained and has no dependency on
PyTorch, so it can be unit-tested on its own (see crypto_utils_selftest()
at the bottom of this file).
"""

import math
import time
from typing import List, Tuple

import numpy as np
from phe import paillier


# ----------------------------------------------------------------------
# Quantization: float deltas <-> signed fixed-point integers
# ----------------------------------------------------------------------
class Quantizer:
    """
    Maps a float32 vector into signed b-bit fixed point integers and back.

    q = round(clip(w, -clip_val, clip_val) * scale)
    q is then restricted to the representable signed range
    [-2^(b-1), 2^(b-1) - 1] by clipping; any value driven outside this
    range by `scale` is clamped (logged as a saturation event so utility
    impact can be reported, not silently hidden).
    """

    def __init__(self, bits: int, scale: float):
        self.bits = bits
        self.scale = scale
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.saturated_count = 0
        self.total_count = 0

    def quantize(self, w: np.ndarray) -> np.ndarray:
        scaled = np.round(w.astype(np.float64) * self.scale)
        self.total_count += scaled.size
        self.saturated_count += int(np.sum((scaled < self.qmin) | (scaled > self.qmax)))
        q = np.clip(scaled, self.qmin, self.qmax).astype(np.int64)
        return q

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return (q.astype(np.float64) / self.scale).astype(np.float32)

    def saturation_rate(self) -> float:
        return 0.0 if self.total_count == 0 else self.saturated_count / self.total_count


# ----------------------------------------------------------------------
# Packing: many signed integers -> few big unsigned "packed" integers
# ----------------------------------------------------------------------
class Packer:
    """
    Packs `slots_per_chunk` signed b-bit integers into one big unsigned
    integer suitable as a single Paillier plaintext, with enough headroom
    per slot (`margin_bits`) that summing `num_clients` packed integers
    cannot let one slot's carry corrupt its neighbour.

    Encoding of a single slot value q (signed, range [qmin, qmax]):
        u = q + bias                      where bias = 2^(bits-1)
    u is therefore unsigned in [0, 2^bits - 1] and is placed in a
    slot_width = bits + margin_bits wide field. After homomorphically
    summing `num_clients` packed plaintexts, slot value lies in
        [0, num_clients * (2^bits - 1)]
    which fits inside slot_width bits as long as
        2^slot_width - 1 >= num_clients * (2^bits - 1).
    """

    def __init__(self, bits: int, num_clients: int, margin_bits: int, modulus_bits: int):
        self.bits = bits
        self.bias = 2 ** (bits - 1)
        needed_margin = math.ceil(math.log2(max(num_clients, 1))) + 1
        self.margin_bits = max(margin_bits, needed_margin)
        self.slot_width = bits + self.margin_bits
        assert (2 ** self.slot_width - 1) >= num_clients * (2 ** bits - 1), (
            "slot_width too small for the requested num_clients; increase margin_bits"
        )
        # Leave generous headroom below the Paillier modulus so the packed
        # plaintext is always < n (required for correct Paillier encryption).
        usable_bits = modulus_bits - 16
        self.slots_per_chunk = max(1, usable_bits // self.slot_width)

    def num_chunks(self, n_values: int) -> int:
        return math.ceil(n_values / self.slots_per_chunk)

    def pack(self, q: np.ndarray) -> List[int]:
        """q: 1-D int64 array of signed quantized values -> list of packed big ints."""
        u = (q + self.bias).astype(object)  # python int objects to avoid overflow
        chunks = []
        for start in range(0, len(u), self.slots_per_chunk):
            block = u[start:start + self.slots_per_chunk]
            packed = 0
            for slot_value in reversed(block.tolist()):
                packed = (packed << self.slot_width) | int(slot_value)
            chunks.append(packed)
        return chunks

    def unpack_sum(self, packed_chunks: List[int], n_values: int, num_clients: int) -> np.ndarray:
        """
        Inverse of pack(), applied to the SUM of `num_clients` packed
        plaintexts. Recovers the summed signed quantized values
        (i.e. sum_i q_i for each parameter position), not yet averaged.
        """
        mask = (1 << self.slot_width) - 1
        out = []
        for chunk in packed_chunks:
            remaining = chunk
            n_in_chunk = min(self.slots_per_chunk, n_values - len(out))
            for _ in range(n_in_chunk):
                slot_sum_unsigned = remaining & mask
                remaining >>= self.slot_width
                out.append(slot_sum_unsigned - num_clients * self.bias)
        return np.array(out[:n_values], dtype=np.int64)


# ----------------------------------------------------------------------
# Paillier wrapper with timing instrumentation (used for the overhead study)
# ----------------------------------------------------------------------
class PaillierAggregator:
    def __init__(self, key_bits: int = 1024):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_bits)
        self.key_bits = key_bits

    def encrypt_chunks(self, chunks: List[int]) -> Tuple[List, float]:
        t0 = time.perf_counter()
        ciphertexts = [self.public_key.encrypt(c) for c in chunks]
        elapsed = time.perf_counter() - t0
        return ciphertexts, elapsed

    @staticmethod
    def homomorphic_sum(client_ciphertext_lists: List[List]) -> Tuple[List, float]:
        """Elementwise homomorphic addition of each client's ciphertext list."""
        t0 = time.perf_counter()
        n_chunks = len(client_ciphertext_lists[0])
        summed = []
        for i in range(n_chunks):
            acc = client_ciphertext_lists[0][i]
            for client in client_ciphertext_lists[1:]:
                acc = acc + client[i]
            summed.append(acc)
        elapsed = time.perf_counter() - t0
        return summed, elapsed

    def decrypt_chunks(self, summed_ciphertexts: List) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        plain = [self.private_key.decrypt(c) for c in summed_ciphertexts]
        elapsed = time.perf_counter() - t0
        return plain, elapsed

    @staticmethod
    def serialized_ciphertext_bytes(ciphertexts: List) -> int:
        """Approximate wire size: each Paillier ciphertext is one big int mod n^2."""
        total_bits = sum(c.ciphertext().bit_length() for c in ciphertexts)
        return math.ceil(total_bits / 8)


# ----------------------------------------------------------------------
# High-level convenience functions used by run_ciphertune.py
# ----------------------------------------------------------------------
def encrypt_delta(vector: np.ndarray, quantizer: Quantizer, packer: Packer,
                   aggregator: PaillierAggregator):
    q = quantizer.quantize(vector)
    chunks = packer.pack(q)
    ciphertexts, enc_time = aggregator.encrypt_chunks(chunks)
    return ciphertexts, enc_time


def aggregate_and_decrypt(client_ciphertexts: List[List], n_values: int, num_clients: int,
                           quantizer: Quantizer, packer: Packer, aggregator: PaillierAggregator):
    summed_ct, sum_time = aggregator.homomorphic_sum(client_ciphertexts)
    summed_plain, dec_time = aggregator.decrypt_chunks(summed_ct)
    summed_q = packer.unpack_sum(summed_plain, n_values, num_clients)
    avg_q = summed_q.astype(np.float64) / num_clients
    avg_delta = quantizer.dequantize(avg_q)
    return avg_delta, sum_time, dec_time


# ----------------------------------------------------------------------
# Self-test: verifies the full pack -> encrypt -> homomorphic-sum ->
# decrypt -> unpack -> dequantize round trip recovers the true average
# of several random client vectors to within quantization error.
# Run directly: `python crypto_utils.py`
# ----------------------------------------------------------------------
def crypto_utils_selftest():
    rng = np.random.default_rng(0)
    n_values = 5000
    num_clients = 8
    bits = 16
    scale = 2 ** 12

    client_vectors = [rng.normal(0, 0.02, size=n_values).astype(np.float32) for _ in range(num_clients)]
    true_avg = np.mean(client_vectors, axis=0)

    quantizer = Quantizer(bits=bits, scale=scale)
    aggregator = PaillierAggregator(key_bits=1024)
    packer = Packer(bits=bits, num_clients=num_clients, margin_bits=4, modulus_bits=aggregator.key_bits)

    print(f"slot_width={packer.slot_width} bits, slots_per_chunk={packer.slots_per_chunk}, "
          f"num_chunks={packer.num_chunks(n_values)}")

    client_cts = []
    total_enc_time = 0.0
    for v in client_vectors:
        ct, enc_time = encrypt_delta(v, quantizer, packer, aggregator)
        client_cts.append(ct)
        total_enc_time += enc_time

    avg_delta, sum_time, dec_time = aggregate_and_decrypt(
        client_cts, n_values, num_clients, quantizer, packer, aggregator
    )

    max_abs_err = float(np.max(np.abs(avg_delta - true_avg)))
    mean_abs_err = float(np.mean(np.abs(avg_delta - true_avg)))
    expected_quant_step = 1.0 / scale

    print(f"total encryption time (8 clients): {total_enc_time:.3f}s")
    print(f"homomorphic sum time:               {sum_time:.6f}s")
    print(f"decryption time:                    {dec_time:.3f}s")
    print(f"max abs error vs true average:      {max_abs_err:.6f}")
    print(f"mean abs error vs true average:      {mean_abs_err:.6f}")
    print(f"expected quantization step size:    {expected_quant_step:.6f}")
    print(f"saturation rate:                     {quantizer.saturation_rate():.6%}")

    assert max_abs_err < 5 * expected_quant_step, "round-trip error exceeds expected quantization bound"
    print("SELF-TEST PASSED: homomorphic packed aggregation reconstructs the true average "
          "within quantization tolerance.")


if __name__ == "__main__":
    crypto_utils_selftest()
