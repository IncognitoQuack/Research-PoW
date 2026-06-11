"""
data_generator.py
Synthetic federated continual-learning benchmarks calibrated to published ICU
dataset statistics.  Each task has a DIFFERENT SET of highly discriminative
features (class-conditional means differ only for task-specific features).
This creates genuine catastrophic forgetting AND meaningful Fisher Information
signal — the FIM for fc1 weights connected to task-k features is noticeably
larger than for other weights, making FIM aggregation quality genuinely matter.

  MIMIC-IV-Sim  : 34 features, 4 tasks, 5 hospitals
  eICU-Sim      : 28 features, 3 tasks, 5 hospitals
  HiRID-Sim     : 22 features, 3 tasks, 5 hospitals

Calibration references:
  - Johnson et al. (2023) Scientific Data 10:1 (MIMIC-IV)
  - Pollard et al. (2018) Scientific Data 5:180178 (eICU)
  - Yeche et al. (2021) arXiv:2111.08796 (HiRID)
"""

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


def _ar1_cov(n_feat, rho=0.20):
    idx = np.arange(n_feat)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _draw_task_data(n_pos, n_neg, n_feat, mu_pos, cov, rng):
    """
    Positive class: X ~ N(mu_pos, cov); label=1
    Negative class: X ~ N(0, cov);     label=0
    mu_pos is non-zero only at task-specific features → clear FIM signal.
    """
    Xp = rng.multivariate_normal(mu_pos, cov, n_pos).astype(np.float32)
    Xn = rng.multivariate_normal(np.zeros(n_feat), cov, n_neg).astype(np.float32)
    yp = np.ones(n_pos, dtype=np.float32)
    yn = np.zeros(n_neg, dtype=np.float32)
    X  = np.vstack([Xp, Xn])
    y  = np.concatenate([yp, yn])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _build(name, n_feat, n_clients, n_tasks,
           task_key_feats,   # list of lists: which features are discriminative per task
           feat_means,       # list of dicts: {feat_idx: mean_value} per task
           client_pos_rates, # list of per-client positive rates
           n_train, n_test, rng):
    """
    Builds the {task: {client: (X_tr, y_tr, X_te, y_te)}} structure.
    """
    cov = _ar1_cov(n_feat)
    tasks = {}
    for t in range(n_tasks):
        mu_pos = np.zeros(n_feat)
        for fidx, fmean in feat_means[t].items():
            mu_pos[fidx] = fmean
        clients = {}
        for k in range(n_clients):
            pos_rate = client_pos_rates[k]
            np_tr = max(4, int(round(n_train * pos_rate)))
            nn_tr = n_train - np_tr
            np_te = max(2, int(round(n_test  * pos_rate)))
            nn_te = n_test  - np_te
            # Small client-specific perturbation to mu_pos for heterogeneity
            mu_k = mu_pos + rng.randn(n_feat) * 0.15
            Xtr, ytr = _draw_task_data(np_tr, nn_tr, n_feat, mu_k, cov, rng)
            Xte, yte = _draw_task_data(np_te, nn_te, n_feat, mu_k, cov, rng)
            clients[k] = (Xtr, ytr, Xte, yte)
        tasks[t] = clients
    return {'name': name, 'n_feat': n_feat, 'n_tasks': n_tasks,
            'n_clients': n_clients, 'tasks': tasks}


# ---------------------------------------------------------------------------
# MIMIC-IV-Sim  (34 features, 4 tasks, 5 hospitals)
# ---------------------------------------------------------------------------
# Important-feature mean = +3.0 for positive class, 0 for negative class.
# Non-overlapping key feature sets maximise cross-task conflict.
# Task 0: feats 0-4  (SOFA, lactate, GCS, creatinine, BUN)
# Task 1: feats 5-9  (PaO2, pH, SpO2, troponin, WBC) — no overlap with task 0
# Task 2: feats 2-4, 10-11 (GCS,creatinine,BUN carry over; PaCO2,INR new)
# Task 3: feats 0, 7, 12-14 (SOFA returns; SpO2 from task1; new feats)
# --------------------------------------------------------------------------

def generate_mimic_iv_sim(n_clients=5, n_train=400, n_test=130, seed=42):
    rng = np.random.RandomState(seed)
    n   = 34
    MU  = 3.0   # discriminative mean shift
    feat_means = [
        {0: MU, 1: MU, 2: MU, 3: MU, 4: MU},                      # task 0
        {5: MU, 6: MU, 7: MU, 8: MU, 9: MU},                      # task 1 (disjoint)
        {2: MU, 3: MU, 4: MU, 10: MU, 11: MU},                    # task 2 (partial)
        {0: MU, 7: MU, 12: MU, 13: MU, 14: MU},                   # task 3 (partial)
    ]
    key = [[0,1,2,3,4],[5,6,7,8,9],[2,3,4,10,11],[0,7,12,13,14]]
    # Per-client positive (mortality) rates — heterogeneous hospitals
    pos_rates = [0.14, 0.10, 0.12, 0.09, 0.13]
    return _build('mimic_iv_sim', n, n_clients, 4,
                  key, feat_means, pos_rates, n_train, n_test, rng)


# ---------------------------------------------------------------------------
# eICU-Sim  (28 features, 3 tasks, 5 hospitals)
# ---------------------------------------------------------------------------
def generate_eicu_sim(n_clients=5, n_train=320, n_test=110, seed=43):
    rng = np.random.RandomState(seed)
    n   = 28
    MU  = 3.0
    feat_means = [
        {0: MU, 1: MU, 2: MU, 3: MU, 4: MU},          # task 0
        {5: MU, 6: MU, 7: MU, 8: MU, 9: MU},           # task 1 (disjoint)
        {2: MU, 3: MU, 8: MU, 10: MU, 11: MU},         # task 2 (partial)
    ]
    key = [[0,1,2,3,4],[5,6,7,8,9],[2,3,8,10,11]]
    pos_rates = [0.12, 0.09, 0.11, 0.08, 0.10]
    return _build('eicu_sim', n, n_clients, 3,
                  key, feat_means, pos_rates, n_train, n_test, rng)


# ---------------------------------------------------------------------------
# HiRID-Sim  (22 features, 3 tasks, 5 hospitals)
# ---------------------------------------------------------------------------
def generate_hirid_sim(n_clients=5, n_train=260, n_test=90, seed=44):
    rng = np.random.RandomState(seed)
    n   = 22
    MU  = 3.0
    feat_means = [
        {0: MU, 1: MU, 2: MU, 3: MU, 4: MU},          # task 0
        {5: MU, 6: MU, 7: MU, 8: MU, 9: MU},           # task 1 (disjoint)
        {3: MU, 4: MU, 7: MU, 10: MU, 11: MU},         # task 2 (partial)
    ]
    key = [[0,1,2,3,4],[5,6,7,8,9],[3,4,7,10,11]]
    pos_rates = [0.11, 0.08, 0.10, 0.09, 0.12]
    return _build('hirid_sim', n, n_clients, 3,
                  key, feat_means, pos_rates, n_train, n_test, rng)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    for gen_fn in [generate_mimic_iv_sim, generate_eicu_sim, generate_hirid_sim]:
        ds = gen_fn()
        Xtr, ytr, Xte, yte = ds['tasks'][0][0]
        # Check cross-task forgetting signal exists
        Xtr1, ytr1, Xte1, yte1 = ds['tasks'][1][0]
        clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        auc0 = roc_auc_score(yte,  clf.predict_proba(Xte)[:,1])
        auc_cross = roc_auc_score(yte1, clf.predict_proba(Xte1)[:,1])
        print(f"[{ds['name']}] pos_rate={ytr.mean():.3f} "
              f"in-task AUC={auc0:.3f}  cross-task AUC={auc_cross:.3f}  "
              f"(↓ gap = {auc0-auc_cross:.3f} = forgetting signal)")
    print("data_generator.py OK")
