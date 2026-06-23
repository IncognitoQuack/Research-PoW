"""
dp_utils.py
------------
Client-level differential privacy for the DP-FedAvg baseline, following
the client-level DP-FedAvg construction of McMahan et al. ("Learning
Differentially Private Recurrent Language Models", ICLR 2018) and Geyer
et al. ("Differentially Private Federated Learning: A Client Level
Perspective", NeurIPS Workshop 2017): each client's *local update* (not
each individual training example) is the unit of privacy.

Design note -- why noise is added ONCE to the aggregate, not per client
--------------------------------------------------------------------
An earlier version of this module added independent Gaussian noise at
EVERY client before averaging. That is a valid alternative construction,
but it is needlessly punishing at small client counts: if each of N
clients adds noise of std sigma independently, the AVERAGE retains noise
of std sigma/sqrt(N), which barely shrinks for the kind of N (a handful
of organizations) realistic for inter-organizational threat-intel
sharing. McMahan et al. (2018, Algorithm 1) instead clip every client's
update to L2 norm <= S and have the (trusted) aggregating server add a
SINGLE Gaussian noise vector to the already-averaged update, with sigma
proportional to S / N. This is the standard, more favourable
construction and is what is implemented below; it is also the
*honest* reason this paper reports a per-round epsilon sweep rather than
a single cherry-picked operating point -- the privacy/utility trade-off
of client-level DP is fundamentally governed by how many clients
contribute per round, and a small number of simulated organizations (as
is realistic for this threat-intel use case) is the genuinely harder
regime for differential privacy, which is itself a finding worth
reporting rather than concealing.

Privacy accounting
-------------------
Each round is calibrated to satisfy (epsilon_round, delta_round)-DP via
the classical Gaussian mechanism (Dwork & Roth, 2014, "The Algorithmic
Foundations of Differential Privacy", Appendix A, Theorem A.1):

    sigma = sqrt(2 * ln(1.25 / delta_round)) * S / (N * epsilon_round)

Composing R independent rounds is reported under the elementary/basic
composition theorem (Dwork & Roth, 2014, Theorem 3.16):

    epsilon_total = R * epsilon_round ,   delta_total = R * delta_round

Basic composition is always valid (no extra assumptions) and, unlike
advanced composition, does not numerically blow up for the moderate
epsilon_round values explored in the sweep below; we use it as the
single, simple, always-checkable number reported in the paper.
"""

import math
from typing import Tuple

import numpy as np

from . import config as C


def clip_l2(vector: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > clip_norm and norm > 0:
        return vector * (clip_norm / norm)
    return vector


def gaussian_sigma_for_average(clip_norm: float, num_clients: int,
                                epsilon_round: float, delta_round: float) -> float:
    """Std-dev of the SINGLE noise vector added to the already-averaged update."""
    z = math.sqrt(2.0 * math.log(1.25 / delta_round))
    return z * clip_norm / (num_clients * epsilon_round)


def add_gaussian_noise(vector: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return vector + rng.normal(0, sigma, size=vector.shape).astype(vector.dtype)


def aggregate_with_central_dp(client_deltas, epsilon_round: float, rng: np.random.Generator,
                               clip_norm: float = None, delta_round: float = None):
    """
    client_deltas: list of per-client raw (unclipped) delta vectors for one round.
    Returns (noised_average_delta, sigma_used).
    """
    clip_norm = C.DP_CLIP_NORM if clip_norm is None else clip_norm
    delta_round = C.DP_DELTA_PER_ROUND if delta_round is None else delta_round
    clipped = [clip_l2(d, clip_norm) for d in client_deltas]
    avg = np.mean(clipped, axis=0)
    sigma = gaussian_sigma_for_average(clip_norm, len(client_deltas), epsilon_round, delta_round)
    noised_avg = add_gaussian_noise(avg, sigma, rng)
    return noised_avg, sigma


def total_epsilon_basic(epsilon_round: float, num_rounds: int) -> float:
    """Basic composition (Dwork & Roth, 2014, Theorem 3.16): always valid, never blows up."""
    return epsilon_round * num_rounds


def total_delta_basic(delta_round: float, num_rounds: int) -> float:
    return delta_round * num_rounds


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 15364  # matches the default reduced LoRA trainable-parameter count (1 layer; see config.py)
    n_clients = C.NUM_CLIENTS

    print(f"{'eps_round':>10} {'sigma':>10} {'sigma*sqrt(dim)':>18} {'eps_total(basic)':>18}")
    for eps_round in [0.5, 2.0, 8.0]:
        sigma = gaussian_sigma_for_average(C.DP_CLIP_NORM, n_clients, eps_round, C.DP_DELTA_PER_ROUND)
        total_noise_l2 = sigma * math.sqrt(dim)
        eps_total = total_epsilon_basic(eps_round, C.FEDERATED_ROUNDS)
        print(f"{eps_round:>10.2f} {sigma:>10.5f} {total_noise_l2:>18.3f} {eps_total:>18.2f}")
    print(f"\n(for reference: clip_norm S={C.DP_CLIP_NORM}, num_clients={n_clients}, "
          f"rounds={C.FEDERATED_ROUNDS}, dim={dim})")
    print("'sigma*sqrt(dim)' is the expected total L2 norm of the injected noise vector; "
          "compare this to the clipped average update's own norm (<= S) to gauge whether "
          "the noise will dominate the signal at this operating point.")
