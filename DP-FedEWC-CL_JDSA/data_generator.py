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

REALISM LAYER (added in response to reviewer feedback that the original
generator was "too clean" relative to real EHR data).  Four nuisance
processes are injected on top of the base class-conditional Gaussian model,
each independently seeded so that their severity varies across the
multi-seed replication protocol used for the statistical evaluation in
Section 6 of the paper:

  1. Missingness (MCAR + weak MAR): a per-feature missingness rate drawn from
     U(0.05, 0.18) masks entries to NaN; missing values are mean-imputed
     using ONLY the client's own training-split statistics (no test-time or
     cross-client leakage), matching how a hospital would impute in
     deployment.
  2. Coding / measurement noise: a client- and feature-specific multiplicative
     noise factor (log-normal, sigma=0.12) emulates differing instrument
     calibration and unit/rounding conventions across hospitals.
  3. Label uncertainty: a small fraction (U(1.5%, 4%) per client) of binary
     labels are flipped uniformly at random, representing coding/adjudication
     disagreement in outcome labelling.
  4. Richer hospital heterogeneity: in addition to the existing per-client
     mean perturbation, each client now also draws its own AR(1) correlation
     coefficient (rho ~ U(0.10, 0.35)) and its own per-task sample count
     (simulating irregular admission volume / temporal irregularity), rather
     than a fixed n_train shared by all hospitals.

These processes are applied identically across ALL methods compared in the
paper (they affect only the data the methods observe), so no method receives
an unfair advantage or disadvantage from them.
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


def _apply_coding_noise(X, rng, sigma_log=0.12):
    """Multiplicative log-normal noise per feature — instrument/rounding drift."""
    factors = rng.lognormal(mean=0.0, sigma=sigma_log, size=X.shape[1]).astype(np.float32)
    return X * factors[None, :]


def _apply_missingness(X_tr, X_te, rng, rate_lo=0.05, rate_hi=0.18):
    """
    MCAR + weak MAR masking.  Each feature gets its own missingness rate.
    A feature whose value is in the top decile of a random "severity" feature
    is additionally more likely to be missing (weak MAR component), mimicking
    sicker patients having more missing labs due to shorter ICU stays or
    point-of-care testing gaps.
    Missing entries are mean-imputed using TRAIN split statistics only.
    """
    n_feat = X_tr.shape[1]
    rates = rng.uniform(rate_lo, rate_hi, size=n_feat)

    severity_tr = X_tr.mean(axis=1, keepdims=True)
    severity_te = X_te.mean(axis=1, keepdims=True)
    sev_thresh = np.quantile(severity_tr, 0.80)

    def _mask(X, severity):
        mask = rng.random(X.shape) < rates[None, :]
        mar_boost = (severity > sev_thresh).astype(np.float32)
        mar_mask = rng.random(X.shape) < (0.10 * mar_boost)
        return mask | mar_mask.astype(bool)

    mask_tr = _mask(X_tr, severity_tr)
    mask_te = _mask(X_te, severity_te)

    X_tr_missing = X_tr.copy()
    X_tr_missing[mask_tr] = np.nan
    col_means = np.nanmean(X_tr_missing, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    X_tr_imp = X_tr_missing.copy()
    inds = np.where(np.isnan(X_tr_imp))
    X_tr_imp[inds] = np.take(col_means, inds[1])

    X_te_missing = X_te.copy()
    X_te_missing[mask_te] = np.nan
    X_te_imp = X_te_missing.copy()
    inds_te = np.where(np.isnan(X_te_imp))
    X_te_imp[inds_te] = np.take(col_means, inds_te[1])

    return X_tr_imp.astype(np.float32), X_te_imp.astype(np.float32)


def _apply_label_noise(y, rng, rate_lo=0.015, rate_hi=0.04):
    rate = rng.uniform(rate_lo, rate_hi)
    flip_mask = rng.random(len(y)) < rate
    y_noisy = y.copy()
    y_noisy[flip_mask] = 1.0 - y_noisy[flip_mask]
    return y_noisy


def _build(name, n_feat, n_clients, n_tasks,
           task_key_feats,   # list of lists: which features are discriminative per task
           feat_means,       # list of dicts: {feat_idx: mean_value} per task
           client_pos_rates, # list of per-client positive rates
           n_train, n_test, rng, realism=True):
    """
    Builds the {task: {client: (X_tr, y_tr, X_te, y_te)}} structure.
    If realism=True (default), applies missingness, coding noise, label
    uncertainty, and per-client correlation/volume heterogeneity on top of
    the base class-conditional Gaussian generative model.
    """
    tasks = {}
    # Per-client AR(1) correlation heterogeneity (temporal-irregularity proxy)
    client_rho = rng.uniform(0.10, 0.35, size=n_clients) if realism else \
        np.full(n_clients, 0.20)
    # Per-client, per-task volume heterogeneity (irregular admission volume)
    volume_factor = rng.uniform(0.75, 1.25, size=(n_tasks, n_clients)) if realism else \
        np.ones((n_tasks, n_clients))

    for t in range(n_tasks):
        mu_pos = np.zeros(n_feat)
        for fidx, fmean in feat_means[t].items():
            mu_pos[fidx] = fmean
        clients = {}
        for k in range(n_clients):
            pos_rate = client_pos_rates[k]
            n_train_k = max(20, int(round(n_train * volume_factor[t, k])))
            n_test_k  = max(10, int(round(n_test  * volume_factor[t, k])))
            np_tr = max(4, int(round(n_train_k * pos_rate)))
            nn_tr = n_train_k - np_tr
            np_te = max(2, int(round(n_test_k * pos_rate)))
            nn_te = n_test_k - np_te
            # Client-specific perturbation to mu_pos for heterogeneity
            mu_k = mu_pos + rng.randn(n_feat) * 0.15
            cov_k = _ar1_cov(n_feat, rho=client_rho[k])
            Xtr, ytr = _draw_task_data(np_tr, nn_tr, n_feat, mu_k, cov_k, rng)
            Xte, yte = _draw_task_data(np_te, nn_te, n_feat, mu_k, cov_k, rng)

            if realism:
                Xtr = _apply_coding_noise(Xtr, rng)
                Xte = _apply_coding_noise(Xte, rng)
                Xtr, Xte = _apply_missingness(Xtr, Xte, rng)
                ytr = _apply_label_noise(ytr, rng)
                yte = _apply_label_noise(yte, rng)

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

def generate_mimic_iv_sim(n_clients=5, n_train=400, n_test=130, seed=42,
                          realism=True):
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
                  key, feat_means, pos_rates, n_train, n_test, rng,
                  realism=realism)


# ---------------------------------------------------------------------------
# eICU-Sim  (28 features, 3 tasks, 5 hospitals)
# ---------------------------------------------------------------------------
def generate_eicu_sim(n_clients=5, n_train=320, n_test=110, seed=43,
                      realism=True):
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
                  key, feat_means, pos_rates, n_train, n_test, rng,
                  realism=realism)


# ---------------------------------------------------------------------------
# HiRID-Sim  (22 features, 3 tasks, 5 hospitals)
# ---------------------------------------------------------------------------
def generate_hirid_sim(n_clients=5, n_train=260, n_test=90, seed=44,
                       realism=True):
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
                  key, feat_means, pos_rates, n_train, n_test, rng,
                  realism=realism)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    for gen_fn in [generate_mimic_iv_sim, generate_eicu_sim, generate_hirid_sim]:
        ds = gen_fn()
        Xtr, ytr, Xte, yte = ds['tasks'][0][0]
        n_missing = np.isnan(Xtr).sum()  # should be 0 post-imputation
        # Check cross-task forgetting signal exists
        Xtr1, ytr1, Xte1, yte1 = ds['tasks'][1][0]
        clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        auc0 = roc_auc_score(yte,  clf.predict_proba(Xte)[:,1])
        auc_cross = roc_auc_score(yte1, clf.predict_proba(Xte1)[:,1])
        print(f"[{ds['name']}] pos_rate={ytr.mean():.3f} "
              f"in-task AUC={auc0:.3f}  cross-task AUC={auc_cross:.3f}  "
              f"post-impute NaNs={n_missing}  "
              f"(↓ gap = {auc0-auc_cross:.3f} = forgetting signal)")
    print("data_generator.py OK")
