"""
Evaluation metrics for cascade misinformation detection.
All functions are pure (no side-effects); model_fn is a callable that accepts
(node_feats: Tensor, adj: Tensor) and returns logits: Tensor of shape (n_classes,).
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true, y_pred, n_classes=4):
    return float(f1_score(y_true, y_pred, average='macro',
                          labels=list(range(n_classes)), zero_division=0))


def compute_auroc(y_true, y_prob):
    """Binary AUROC: rumour (label > 0) vs non-rumour (label == 0)."""
    bin_true   = (np.array(y_true) > 0).astype(int)
    rumour_prob = 1.0 - np.array(y_prob)[:, 0]   # prob of being a rumour
    if len(np.unique(bin_true)) < 2:
        return 0.5
    return float(roc_auc_score(bin_true, rumour_prob))


def _truncate_cascade(cascade, k):
    """Return a sub-cascade using the earliest k nodes by timestamp."""
    from data_generator import cascade_to_tensors
    n     = cascade['n_nodes']
    k     = max(2, k)
    order = np.argsort(cascade['timestamps'])[:k]
    o_set = set(order.tolist())
    imap  = {orig: new for new, orig in enumerate(order.tolist())}
    sub   = {
        'node_feats': cascade['node_feats'][order],
        'edges':      [(imap[s], imap[d])
                       for (s, d) in cascade['edges']
                       if s in o_set and d in o_set],
        'timestamps': cascade['timestamps'][order],
        'label':      cascade['label'],
        'n_nodes':    k,
        'max_depth':  0,
    }
    return cascade_to_tensors(sub)


def compute_eda(model_fn, cascades, threshold=0.20, device='cpu'):
    """
    Early Detection Accuracy: accuracy when only the first `threshold` fraction
    of posts (by arrival timestamp) are observed.
    """
    preds, labels = [], []
    for cas in cascades:
        k    = max(2, int(np.ceil(threshold * cas['n_nodes'])))
        feats, adj = _truncate_cascade(cas, k)
        feats, adj = feats.to(device), adj.to(device)
        with torch.no_grad():
            logits = model_fn(feats, adj)
        preds.append(int(logits.argmax().item()))
        labels.append(cas['label'])
    return float(accuracy_score(labels, preds))


def compute_mdl(model_fn, cascades, cascade_minutes, conf_thresh=0.65, device='cpu'):
    """
    Mean Detection Lag (minutes): mean time from cascade start to the first
    correct, high-confidence prediction.  Sweeps 10 % increments; if the model
    never achieves confidence ≥ conf_thresh on the correct class within the
    full cascade, the full cascade duration is charged.
    """
    steps = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    lags  = []
    for cas in cascades:
        true_lbl     = cas['label']
        detected_frac = 1.0
        for frac in steps:
            k = max(2, int(np.ceil(frac * cas['n_nodes'])))
            feats, adj = _truncate_cascade(cas, k)
            feats, adj = feats.to(device), adj.to(device)
            with torch.no_grad():
                logits = model_fn(feats, adj)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            if int(probs.argmax()) == true_lbl and float(probs.max()) >= conf_thresh:
                detected_frac = frac
                break
        lags.append(detected_frac * cascade_minutes)
    return float(np.mean(lags))
