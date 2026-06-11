"""
metrics.py
All evaluation metrics used in the paper.

  avg_auc     — Mean AUROC across all T tasks (R_matrix last row)
  bwt         — Backward Transfer: negative values indicate forgetting
  fwt         — Forward Transfer: positive values indicate knowledge benefit
  avg_acc     — Mean balanced accuracy across all tasks
  privacy_eps — Cumulative (ε, δ)-DP budget (passed in from the runner)

BWT / FWT follow the definitions of Lopez-Paz & Ranzato (2017) and
Chaudhry et al. (2018), which are standard in the continual-learning
literature.
"""

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

np.random.seed(42)


# ---------------------------------------------------------------------------
# Per-sample evaluation helpers (called by the runner, not the metrics below)
# ---------------------------------------------------------------------------

def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Handles degenerate (single-class) test splits gracefully."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(balanced_accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Continual-learning metrics from R_matrix
# ---------------------------------------------------------------------------
# Convention: R_matrix[t, j] = performance on task j *after* training on
# tasks 0..t.  Shape: (n_tasks, n_tasks).  Values in [0, 1].

def compute_avg_auc(R_matrix: np.ndarray) -> float:
    """
    Average AUROC across all tasks evaluated after the final training step.
    avg_auc = (1/T) * sum_j R[T-1, j]
    """
    T = R_matrix.shape[0]
    return float(np.mean(R_matrix[T - 1, :]))


def compute_bwt(R_matrix: np.ndarray) -> float:
    """
    Backward Transfer:
      BWT = (1 / (T-1)) * sum_{j=0}^{T-2} [ R[T-1, j] - R[j, j] ]

    Negative BWT indicates catastrophic forgetting (performance on past
    tasks has degraded since they were first trained on).
    """
    T = R_matrix.shape[0]
    if T < 2:
        return 0.0
    diffs = [R_matrix[T - 1, j] - R_matrix[j, j] for j in range(T - 1)]
    return float(np.mean(diffs))


def compute_fwt(R_matrix: np.ndarray, r_random: float = 0.5) -> float:
    """
    Forward Transfer:
      FWT = (1 / (T-1)) * sum_{j=1}^{T-1} [ R[j-1, j] - r_random ]

    R[j-1, j] is the performance on task j *before* any training on that
    task (zero-shot transfer from the model trained on tasks 0..j-1).
    r_random = 0.5 is the expected AUC of a random binary classifier.
    Positive FWT indicates knowledge benefit from previous tasks.
    """
    T = R_matrix.shape[0]
    if T < 2:
        return 0.0
    diffs = [R_matrix[j - 1, j] - r_random for j in range(1, T)]
    return float(np.mean(diffs))


def compute_avg_acc(R_matrix: np.ndarray) -> float:
    """
    Average accuracy in the sense of Lopez-Paz & Ranzato (2017):
      avg_acc = (1/T) * sum_j R[T-1, j]
    (Identical to avg_auc when AUC is the performance measure; retained
    as a separate function for clarity in tables.)
    """
    return compute_avg_auc(R_matrix)


def compute_all_metrics(R_matrix: np.ndarray,
                        privacy_epsilon: float,
                        latency_ms: float = 0.0) -> dict:
    """
    Compute the full set of metrics reported in Table 2 of the paper.
    Returns a dict with float values (serialisable to JSON).
    """
    return {
        'avg_auc':           round(compute_avg_auc(R_matrix), 4),
        'bwt':               round(compute_bwt(R_matrix),     4),
        'fwt':               round(compute_fwt(R_matrix),     4),
        'avg_acc':           round(compute_avg_acc(R_matrix), 4),
        'privacy_epsilon':   round(privacy_epsilon, 3)
                                if np.isfinite(privacy_epsilon) else 99.999,
        'latency_median_ms': round(latency_ms, 2),
    }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    # Construct a synthetic R_matrix that represents mild forgetting
    T = 4
    R = np.array([
        [0.78, 0.62, 0.60, 0.57],   # after task 0
        [0.72, 0.79, 0.61, 0.58],   # after task 1
        [0.71, 0.76, 0.80, 0.62],   # after task 2
        [0.69, 0.74, 0.78, 0.81],   # after task 3
    ])

    m = compute_all_metrics(R, privacy_epsilon=3.21, latency_ms=31.4)
    print("Metrics from synthetic R_matrix:")
    for k, v in m.items():
        print(f"  {k:25s} = {v}")

    # Verify BWT by hand: (R[3,0]-R[0,0] + R[3,1]-R[1,1] + R[3,2]-R[2,2]) / 3
    expected_bwt = ((0.69 - 0.78) + (0.74 - 0.79) + (0.78 - 0.80)) / 3
    assert abs(m['bwt'] - expected_bwt) < 1e-4, "BWT mismatch"
    print("metrics.py OK — all checks passed")
