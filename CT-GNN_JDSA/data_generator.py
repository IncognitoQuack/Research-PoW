"""
Synthetic Heterogeneous Edge-IoT Data Generator
================================================
Generates three benchmark-like datasets mimicking:
  - Dataset A  (SWaT-style)  : industrial water-treatment, 51 sensor nodes
  - Dataset B  (WADI-style)  : water-distribution, 127 sensor nodes
  - Dataset C  (PSM-style)   : pooled edge-server metrics, 25 nodes

Each dataset embeds four node types (temperature, pressure, flow, network)
and five anomaly categories with planted Granger-causal propagation chains,
so that root-cause localisation can be evaluated deterministically.

Seed: 42 throughout for reproducibility.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    name: str
    n_nodes: int
    n_train: int
    n_val: int
    n_test: int
    anomaly_rate: float          # fraction of test timesteps that are anomalous
    n_anomaly_events: int        # distinct anomaly injection events
    node_types: List[str]        # cycling list of node-type labels
    causal_density: float        # edge density in planted causal graph
    drift_start: int             # timestep in test split where concept drift begins


CONFIGS = {
    "SWaT": DatasetConfig(
        name="SWaT",
        n_nodes=51,
        n_train=40000,
        n_val=5000,
        n_test=15000,
        anomaly_rate=0.122,
        n_anomaly_events=36,
        node_types=["temperature","pressure","flow","network"],
        causal_density=0.08,
        drift_start=8000,
    ),
    "WADI": DatasetConfig(
        name="WADI",
        n_nodes=127,
        n_train=70000,
        n_val=8000,
        n_test=17280,
        anomaly_rate=0.057,
        n_anomaly_events=15,
        node_types=["temperature","pressure","flow","network","power"],
        causal_density=0.04,
        drift_start=10000,
    ),
    "PSM": DatasetConfig(
        name="PSM",
        n_nodes=25,
        n_train=87841,
        n_val=10000,
        n_test=43901,
        anomaly_rate=0.276,
        n_anomaly_events=120,
        node_types=["cpu","memory","network","disk"],
        causal_density=0.12,
        drift_start=25000,
    ),
}


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def _make_causal_adj(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """
    Build a directed acyclic graph (DAG) adjacency matrix with given edge
    density.  DAG property is ensured by only allowing edges i -> j where i < j
    in a topologically sorted node ordering (random permutation).
    """
    perm = rng.permutation(n)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                src, dst = perm[i], perm[j]
                A[src, dst] = rng.uniform(0.3, 0.9)
    return A


def _var_process(n: int, T: int, A: np.ndarray,
                 rng: np.random.Generator, lag: int = 3) -> np.ndarray:
    """
    Simulate a vector auto-regressive (VAR) process with causal structure A,
    returning array of shape (T, n).  Innovation noise is heteroskedastic to
    simulate different sensor types.
    """
    noise_scales = rng.uniform(0.05, 0.3, size=n)
    X = rng.standard_normal((T + lag, n)) * noise_scales
    for t in range(lag, T + lag):
        causal_contrib = np.zeros(n)
        for l in range(1, lag + 1):
            # Each node j contributes to node i if A[i,j] > 0
            causal_contrib += A @ X[t - l]  # shape (n,)
        X[t] += 0.4 * causal_contrib + rng.standard_normal(n) * noise_scales
    return X[lag:]   # (T, n)


def _inject_anomalies(X: np.ndarray, A: np.ndarray,
                      config: DatasetConfig,
                      rng: np.random.Generator
                      ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Inject anomalies into X (test split only, shape (T_test, n)).
    Returns:
        X_anom     : anomaly-injected array
        labels     : binary array (T_test,), 1 where anomalous
        root_causes: dict mapping anomaly_event_id -> root_cause_node_index
    """
    T, n = X.shape
    X_anom = X.copy()
    labels = np.zeros(T, dtype=np.int32)
    root_causes: Dict[int, int] = {}

    # Identify nodes with highest out-degree as plausible root causes
    out_degree = (A > 0).sum(axis=1)
    high_degree = np.argsort(out_degree)[::-1][:max(5, n // 8)]

    event_lengths = rng.integers(30, 200, size=config.n_anomaly_events)
    used: List[Tuple[int,int]] = []

    for ev_id in range(config.n_anomaly_events):
        # Find a non-overlapping start
        length = int(event_lengths[ev_id])
        for _ in range(200):
            start = int(rng.integers(0, T - length))
            end = start + length
            overlap = any(s < end and e > start for s, e in used)
            if not overlap:
                break
        else:
            continue

        used.append((start, end))
        rc = int(rng.choice(high_degree))
        root_causes[ev_id] = rc

        # Anomaly type (cycle through 5 types)
        atype = ev_id % 5
        # Find causal descendants
        descendants = np.where(A[rc] > 0)[0].tolist()

        affected = [rc] + descendants[:min(len(descendants), 4)]

        for t in range(start, end):
            if atype == 0:   # spike
                for idx in affected:
                    X_anom[t, idx] += rng.uniform(3, 6) * X.std(axis=0)[idx]
            elif atype == 1: # dip
                for idx in affected:
                    X_anom[t, idx] -= rng.uniform(2, 5) * X.std(axis=0)[idx]
            elif atype == 2: # noise injection
                for idx in affected:
                    X_anom[t, idx] += rng.normal(0, 3 * X.std(axis=0)[idx])
            elif atype == 3: # trend shift
                frac = (t - start) / length
                for idx in affected:
                    X_anom[t, idx] += frac * 4 * X.std(axis=0)[idx]
            else:            # plateau (stuck-at)
                for idx in affected:
                    X_anom[t, idx] = X[start, idx] + rng.uniform(-0.05, 0.05)

        labels[start:end] = 1

    return X_anom, labels, root_causes


def _add_concept_drift(X: np.ndarray, drift_start: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Add concept drift by gradually shifting the mean and variance of all
    nodes after drift_start, simulating changing operating conditions.
    """
    T, n = X.shape
    X_drift = X.copy()
    drift_amplitude = rng.uniform(0.3, 0.8, size=n)
    drift_var_factor = rng.uniform(1.2, 2.0, size=n)
    for t in range(drift_start, T):
        alpha = min(1.0, (t - drift_start) / max(1, T - drift_start))
        X_drift[t] = (X[t] * (drift_var_factor ** alpha)
                      + alpha * drift_amplitude)
    return X_drift


def generate_dataset(config_name: str) -> Dict:
    """
    Full dataset generation pipeline.

    Returns a dictionary with:
        train_X  : (n_train, n_nodes)
        val_X    : (n_val,   n_nodes)
        test_X   : (n_test,  n_nodes)  — anomaly injected + concept drift
        test_y   : (n_test,)            — binary anomaly labels
        causal_adj : (n_nodes, n_nodes) — planted causal adjacency
        root_causes : dict {event_id -> root_node}
        node_types  : list of str, length n_nodes
        config      : DatasetConfig
    """
    cfg = CONFIGS[config_name]
    rng = np.random.default_rng(42)

    # 1. Build causal adjacency
    A = _make_causal_adj(cfg.n_nodes, cfg.causal_density, rng)

    # 2. Simulate VAR process
    T_total = cfg.n_train + cfg.n_val + cfg.n_test
    X_full = _var_process(cfg.n_nodes, T_total, A, rng)

    # Standardise
    mu = X_full[:cfg.n_train].mean(axis=0, keepdims=True)
    sigma = X_full[:cfg.n_train].std(axis=0, keepdims=True) + 1e-6
    X_full = (X_full - mu) / sigma

    # 3. Split
    X_train = X_full[:cfg.n_train]
    X_val   = X_full[cfg.n_train: cfg.n_train + cfg.n_val]
    X_test  = X_full[cfg.n_train + cfg.n_val:]

    # 4. Inject anomalies into test split
    X_test_anom, labels, root_causes = _inject_anomalies(
        X_test, A, cfg, rng)

    # 5. Add concept drift into test split
    X_test_final = _add_concept_drift(X_test_anom, cfg.drift_start, rng)

    # 6. Node type assignment (cycling)
    n_types = len(cfg.node_types)
    node_type_list = [cfg.node_types[i % n_types] for i in range(cfg.n_nodes)]

    print(f"[{config_name}] n_nodes={cfg.n_nodes}  "
          f"train={cfg.n_train}  val={cfg.n_val}  test={cfg.n_test}  "
          f"anomaly_rate={labels.mean():.3f}  "
          f"causal_edges={int((A > 0).sum())}")

    return {
        "train_X": X_train.astype(np.float32),
        "val_X":   X_val.astype(np.float32),
        "test_X":  X_test_final.astype(np.float32),
        "test_y":  labels,
        "causal_adj": A,
        "root_causes": root_causes,
        "node_types": node_type_list,
        "config": cfg,
    }


if __name__ == "__main__":
    for name in ["SWaT", "WADI", "PSM"]:
        d = generate_dataset(name)
        print(f"  causal_adj shape: {d['causal_adj'].shape}, "
              f"root_cause events: {len(d['root_causes'])}")
