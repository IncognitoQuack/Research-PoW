"""
Granger-Causality-Informed Graph Construction
==============================================
Implements the graph construction module of CT-GNN.  For each ordered pair
of nodes (i, j) the module tests whether the past values of node j carry
statistically significant predictive information about node i, over and above
what is already provided by i's own past.  This is the Granger (1969)
definition of causality operationalised through vector auto-regression.

To keep inference-time cost tractable on edge hardware, we use a lag-limited
VAR formulation (max_lag ≤ 5) and the F-statistic significance threshold
instead of iterative likelihood-ratio tests.

Reference:
  Granger, C.W.J. (1969). Investigating causal relations by econometric
  models and cross-spectral methods. Econometrica 37(3), 424–438.
  https://doi.org/10.2307/1912791
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional


def _ols_rss(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Ordinary-least-squares regression of Y on X; returns residual sum
    of squares.  Uses the normal equations for speed on small problems.
    """
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        return float(np.dot(resid, resid))
    except np.linalg.LinAlgError:
        return float(np.dot(Y, Y))   # fallback: intercept-only RSS


def pairwise_granger(data: np.ndarray,
                     max_lag: int = 3,
                     alpha: float = 0.05,
                     subsample: int = 5000) -> np.ndarray:
    """
    Compute pairwise Granger-causality adjacency matrix.

    Parameters
    ----------
    data      : array of shape (T, N) — standardised multivariate time series
    max_lag   : number of lags in the VAR restricted / unrestricted models
    alpha     : significance threshold for the F-test (default 0.05)
    subsample : use at most `subsample` rows for speed (random draw, seed 0)

    Returns
    -------
    A : float32 array of shape (N, N); A[i, j] = F-statistic if j
        Granger-causes i at level alpha, else 0.
    """
    T, N = data.shape
    rng = np.random.default_rng(0)

    # Subsample if data is very long
    if T > subsample:
        idx = np.sort(rng.choice(T - max_lag, size=subsample, replace=False))
    else:
        idx = np.arange(T - max_lag)

    # Build lagged feature matrix rows indexed by idx
    def _build_lagged(series_2d: np.ndarray,
                      col_indices,
                      t_rows) -> np.ndarray:
        """
        Construct design matrix of shape (len(t_rows), max_lag * len(col_indices) + 1).
        Rows are [1, x_{t-1,cols}, x_{t-2,cols}, ..., x_{t-max_lag,cols}].
        """
        parts = [np.ones((len(t_rows), 1))]
        for lag in range(1, max_lag + 1):
            parts.append(series_2d[t_rows - lag + len(t_rows) - len(t_rows)]
                         if False else series_2d[idx + (T - T), :][:, col_indices] * 0)
        # Re-implement cleanly
        rows = []
        for lag in range(1, max_lag + 1):
            rows.append(series_2d[t_rows + max_lag - lag][:, col_indices])
        return np.hstack([np.ones((len(t_rows), 1))] + rows)

    # Pre-build the full lagged matrix for all nodes
    # Response at t, predictors from t-1..t-max_lag
    t_rows = idx                            # time indices for responses
    Y_all = data[t_rows + max_lag]          # (M, N)

    # Restricted (auto-regressive) design matrix for each node i:
    # uses only lags of node i
    # Unrestricted: uses lags of node i AND lags of node j

    A = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # Build restricted model for node i
        X_restricted_parts = [np.ones((len(t_rows), 1))]
        for lag in range(1, max_lag + 1):
            X_restricted_parts.append(data[t_rows + max_lag - lag, i:i+1])
        X_r = np.hstack(X_restricted_parts)       # (M, max_lag+1)
        Y_i = Y_all[:, i]                          # (M,)
        RSS_r = _ols_rss(Y_i, X_r)
        df_r = len(t_rows) - X_r.shape[1]         # residual df restricted

        for j in range(N):
            if i == j:
                continue
            # Unrestricted model: restricted + lags of j
            X_add_parts = []
            for lag in range(1, max_lag + 1):
                X_add_parts.append(data[t_rows + max_lag - lag, j:j+1])
            X_u = np.hstack([X_r] + X_add_parts)   # (M, max_lag + 1 + max_lag)
            RSS_u = _ols_rss(Y_i, X_u)
            df_u = len(t_rows) - X_u.shape[1]

            if df_u <= 0 or RSS_u <= 0:
                continue

            # F-statistic
            F = ((RSS_r - RSS_u) / max_lag) / (RSS_u / df_u + 1e-10)
            p_val = scipy_stats.f.sf(F, max_lag, df_u)

            if p_val < alpha:
                A[i, j] = float(F)   # strength = F-statistic value

    return A


def threshold_adjacency(A_raw: np.ndarray,
                        top_k: Optional[int] = None,
                        percentile: float = 75.0) -> np.ndarray:
    """
    Binarise or sparsify the raw F-statistic matrix.

    Parameters
    ----------
    A_raw     : (N, N) float matrix from pairwise_granger
    top_k     : keep top-k edges per node (column-wise); if None use percentile
    percentile: keep edges above this percentile of non-zero entries

    Returns
    -------
    A_bin : binary float32 adjacency matrix (N, N)
    """
    A_bin = np.zeros_like(A_raw, dtype=np.float32)
    nonzero = A_raw[A_raw > 0]
    if len(nonzero) == 0:
        return A_bin

    if top_k is not None:
        for j in range(A_raw.shape[1]):
            col = A_raw[:, j].copy()
            col[j] = 0
            if col.max() > 0:
                threshold = np.sort(col)[::-1][min(top_k - 1, (col > 0).sum() - 1)]
                A_bin[:, j] = (col >= threshold).astype(np.float32)
    else:
        thr = np.percentile(nonzero, percentile)
        A_bin = (A_raw >= thr).astype(np.float32)
        np.fill_diagonal(A_bin, 0)

    return A_bin


if __name__ == "__main__":
    # Quick sanity check with planted causal structure
    rng = np.random.default_rng(99)
    T, N = 600, 6
    X = rng.standard_normal((T, N))
    # Plant causality: node 0 causes node 2, node 1 causes node 3
    for t in range(3, T):
        X[t, 2] += 0.6 * X[t-1, 0]
        X[t, 3] += 0.5 * X[t-2, 1]

    A_raw = pairwise_granger(X, max_lag=3, alpha=0.05, subsample=500)
    A_bin = threshold_adjacency(A_raw, percentile=80)
    print("Raw F-stats (selected):")
    print(f"  A[2,0]={A_raw[2,0]:.2f}  (should be high — planted)")
    print(f"  A[3,1]={A_raw[3,1]:.2f}  (should be high — planted)")
    print(f"  A[0,2]={A_raw[0,2]:.2f}  (should be ~0   — no reverse)")
    print(f"Binary edges detected: {int(A_bin.sum())}")
