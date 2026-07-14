"""
data_generator_longhorizon.py
Extended-horizon variant of MIMIC-IV-Sim with T=8 tasks (two full cycles of
the original 4-task drift pattern, each cycle re-randomised in its exact
feature means and per-client heterogeneity). Used solely for the
long-sequence budget-recycling experiment in Section 6.6: budget recycling
reinvests residual epsilon from early-converging tasks into later ones, so
its benefit should compound as the number of tasks grows. A 4-task sequence
gives the recycling mechanism only 3 opportunities to compound; an 8-task
sequence gives it 7, which is the regime in which the mechanism is designed
to matter most.

This benchmark reuses the same generative family (class-conditional
Gaussian with AR(1) covariance, missingness, coding noise, label
uncertainty, and per-client/task heterogeneity) as data_generator.py, so it
is directly comparable to MIMIC-IV-Sim, only longer.
"""

import numpy as np

from data_generator import (
    _build,
)


def generate_mimic_iv_sim_long(n_clients=5, n_train=400, n_test=130,
                               seed=42, n_tasks=8, realism=True):
    rng = np.random.RandomState(seed)
    n = 34
    MU = 3.0

    # Base 4-task drift pattern, repeated for ceil(n_tasks/4) cycles with a
    # small random re-shuffling of the "returning" feature subset each cycle
    # so that cycle 2 is not an exact repeat of cycle 1.
    base_feat_sets = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [2, 3, 4, 10, 11],
        [0, 7, 12, 13, 14],
    ]
    feat_means = []
    key = []
    for t in range(n_tasks):
        cycle = t // 4
        base_idx = t % 4
        feats = list(base_feat_sets[base_idx])
        if cycle > 0:
            # Perturb which auxiliary feature rotates in, so later cycles
            # are related to but not identical to earlier ones.
            extra = int(rng.choice([15, 16, 17, 18, 19, 20]))
            feats = feats[:-1] + [extra]
        feat_means.append({f: MU for f in feats})
        key.append(feats)

    pos_rates = [0.14, 0.10, 0.12, 0.09, 0.13]
    return _build('mimic_iv_sim_long', n, n_clients, n_tasks,
                  key, feat_means, pos_rates, n_train, n_test, rng,
                  realism=realism)


if __name__ == '__main__':
    ds = generate_mimic_iv_sim_long(seed=42)
    print(f"n_tasks={ds['n_tasks']}  n_feat={ds['n_feat']}  "
          f"n_clients={ds['n_clients']}")
    print("data_generator_longhorizon.py OK")
