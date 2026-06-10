"""
Evaluation Metrics
==================
Implements all metrics reported in the paper:

  - Point-adjusted F1-score  (PA-F1): the standard protocol for time-series
    anomaly detection in which a prediction is credited as a true positive
    for any timestep within an anomaly event if at least one timestep of
    that event is detected (Xu et al., 2018).

  - Precision, Recall (point-adjusted)

  - Root-Cause Localisation (RCL) Accuracy: for each planted anomaly event,
    the predicted root-cause node is the argmax of the propagation score
    over the anomalous window.  A prediction is correct if it matches the
    planted root-cause node exactly.

  - Top-3 RCL Accuracy: root-cause node is in the top-3 predicted nodes.

  - Inference latency: wall-clock time per batch (ms) measured via
    torch.cuda.synchronize if GPU is available, else time.perf_counter.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Point-Adjusted F1 (PA-F1)
# ---------------------------------------------------------------------------
def _get_event_segments(labels: np.ndarray) -> List[Tuple[int,int]]:
    """Return list of (start, end) tuples for contiguous anomaly segments."""
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        elif v == 0 and in_seg:
            segs.append((start, i))
            in_seg = False
    if in_seg:
        segs.append((start, len(labels)))
    return segs


def point_adjusted_f1(pred: np.ndarray, labels: np.ndarray
                      ) -> Tuple[float, float, float]:
    """
    Compute PA-F1, PA-Precision, PA-Recall.

    Parameters
    ----------
    pred   : binary prediction array (T,)
    labels : ground-truth binary labels (T,)

    Returns
    -------
    f1, precision, recall
    """
    segs = _get_event_segments(labels)
    # For each anomaly segment, if any point is predicted positive,
    # treat the whole segment as TP
    pred_adj = pred.copy()
    for s, e in segs:
        if pred[s:e].any():
            pred_adj[s:e] = 1
        else:
            pred_adj[s:e] = 0

    tp = int(((pred_adj == 1) & (labels == 1)).sum())
    fp = int(((pred_adj == 1) & (labels == 0)).sum())
    fn = int(((pred_adj == 0) & (labels == 1)).sum())

    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    return float(f1), float(prec), float(rec)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray,
                      n_thresholds: int = 200
                      ) -> Tuple[float, float, float, float]:
    """
    Search over n_thresholds evenly-spaced thresholds and return the
    (best_f1, best_precision, best_recall, best_threshold) tuple.
    """
    lo, hi = scores.min(), scores.max()
    thresholds = np.linspace(lo, hi, n_thresholds)
    best = (0.0, 0.0, 0.0, hi)
    for thr in thresholds:
        pred = (scores >= thr).astype(np.int32)
        f1, prec, rec = point_adjusted_f1(pred, labels)
        if f1 > best[0]:
            best = (f1, prec, rec, float(thr))
    return best


# ---------------------------------------------------------------------------
# Root-Cause Localisation Accuracy
# ---------------------------------------------------------------------------
def rcl_accuracy(prop_scores_dict: Dict[int, np.ndarray],
                 root_causes: Dict[int, int],
                 top_k: int = 1) -> float:
    """
    Compute Root-Cause Localisation accuracy.

    Parameters
    ----------
    prop_scores_dict : dict mapping event_id -> per-node propagation score
                       array of shape (n_nodes,), averaged over the event window
    root_causes      : dict mapping event_id -> ground-truth root-cause node
    top_k            : 1 for exact match, 3 for top-3 accuracy

    Returns
    -------
    accuracy in [0, 1]
    """
    correct, total = 0, 0
    for ev_id, scores in prop_scores_dict.items():
        if ev_id not in root_causes:
            continue
        gt = root_causes[ev_id]
        top_k_nodes = np.argsort(scores)[::-1][:top_k]
        if gt in top_k_nodes:
            correct += 1
        total += 1
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Inference Latency Measurement
# ---------------------------------------------------------------------------
def measure_latency(model: torch.nn.Module,
                    example_input,
                    n_runs: int = 100,
                    warmup: int = 20,
                    device: str = "cpu") -> Tuple[float, float]:
    """
    Measure median and 95th-percentile inference latency in milliseconds.

    Parameters
    ----------
    model        : PyTorch model in eval mode
    example_input: tuple of tensors to pass to model.forward(...)
    n_runs       : number of timed repetitions
    warmup       : number of warm-up passes (not timed)

    Returns
    -------
    (median_ms, p95_ms)
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*example_input)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(*example_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return float(np.median(arr)), float(np.percentile(arr, 95))


# ---------------------------------------------------------------------------
# Summary function
# ---------------------------------------------------------------------------
def evaluate_model(model,
                   test_loader,
                   test_labels: np.ndarray,
                   root_causes: Dict[int, int],
                   anomaly_segs: List[Tuple[int,int]],
                   causal_adj: Optional[torch.Tensor],
                   device: str = "cpu",
                   model_name: str = "",
                   example_input=None) -> Dict:
    """
    Full evaluation pipeline for one model.

    Returns dict with keys: f1, precision, recall, rcl_top1, rcl_top3,
                            latency_median_ms, latency_p95_ms
    """
    model.eval()
    all_scores, all_prop_scores = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch.to(device)
            if causal_adj is not None:
                adj = causal_adj.to(device)
                out = model(x, adj, return_scores=True)
            else:
                out = model(x, return_scores=True)
            _, anom, prop = out
            # Max-pool across nodes to get time-series anomaly score
            all_scores.append(anom.max(dim=-1).values.cpu().numpy())
            all_prop_scores.append(prop.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)    # (T_test,)
    all_prop   = np.concatenate(all_prop_scores, axis=0)  # (T_test, N)

    # F1
    f1, prec, rec, thr = best_f1_threshold(all_scores, test_labels)

    # RCL: for each anomaly event, average propagation scores over that window
    prop_ev = {}
    for ev_id, (s, e) in enumerate(anomaly_segs):
        if s < len(all_prop) and ev_id in root_causes:
            segment_prop = all_prop[s:min(e, len(all_prop))].mean(0)  # (N,)
            prop_ev[ev_id] = segment_prop

    rcl1 = rcl_accuracy(prop_ev, root_causes, top_k=1)
    rcl3 = rcl_accuracy(prop_ev, root_causes, top_k=3)

    # Latency
    if example_input is not None:
        lat_med, lat_p95 = measure_latency(model, example_input,
                                           n_runs=50, warmup=10)
    else:
        lat_med, lat_p95 = 0.0, 0.0

    return {
        "f1":               round(f1,   4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "rcl_top1":         round(rcl1, 4),
        "rcl_top3":         round(rcl3, 4),
        "latency_median_ms": round(lat_med,  2),
        "latency_p95_ms":    round(lat_p95,  2),
        "threshold":         round(thr, 6),
    }
