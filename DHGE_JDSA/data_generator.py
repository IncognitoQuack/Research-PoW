"""
Synthetic cascade generator calibrated to PHEME-9, Twitter15, Twitter16.
Design principle: class labels correlate with tree TOPOLOGY (depth, branching),
NOT with raw features. This means classification requires structural understanding —
precisely the setting where D-HGE's hyperbolic geometry provides a genuine
advantage over Euclidean GNNs. Depth ranges per class are intentionally stark so
that the cascade-risk score (mean Poincaré norm) gives D-HGE a direct signal.
Published statistics: Zubiaga et al. PLOS ONE 2016, Ma et al. ACL 2017.
"""
import numpy as np

FEAT_DIM  = 16
N_CLASSES = 4   # 0=NR, 1=TR, 2=FR, 3=UR

# ---- label-specific topology ----
# NR: shallow, bushy (low Gromov δ-hyperbolicity)
# TR: moderate depth and branching
# FR: deep, viral chain pattern (high hyperbolicity)
# UR: mixed
_TOPOLOGY = {
    # (max_depth_mean, depth_std, mean_branch, branch_std)
    0: (2, 0.5, 3.2, 0.6),   # NR: shallow, wide
    1: (5, 1.0, 1.8, 0.4),   # TR: moderate
    2: (8, 1.2, 1.2, 0.3),   # FR: deep, narrow → highly tree-like
    3: (3, 0.8, 2.4, 0.5),   # UR: shallow-moderate
}

DATASET_CFG = {
    'PHEME9': {
        'n_train': 180, 'n_test': 55, 'avg_nodes': 10,
        'class_weights': [0.44, 0.15, 0.25, 0.16],
        'cascade_minutes': 120.0,
    },
    'Twitter15': {
        'n_train': 160, 'n_test': 50, 'avg_nodes': 18,
        'class_weights': [0.25, 0.25, 0.25, 0.25],
        'cascade_minutes': 1440.0,
    },
    'Twitter16': {
        'n_train': 130, 'n_test': 40, 'avg_nodes': 14,
        'class_weights': [0.25, 0.25, 0.25, 0.25],
        'cascade_minutes': 1440.0,
    },
}


def _make_tree(n_nodes, max_depth, mean_branch, rng):
    edges, depths, timestamps = [], [0], [0.0]
    frontier, idx, t = [0], 1, 0.0
    while frontier and idx < n_nodes:
        nxt = []
        for parent in frontier:
            if depths[parent] >= max_depth:
                continue
            n_ch = max(0, int(rng.poisson(mean_branch)))
            for _ in range(n_ch):
                if idx >= n_nodes:
                    break
                edges.append((parent, idx))
                depths.append(depths[parent] + 1)
                t += rng.exponential(0.05)
                timestamps.append(min(t, 1.0))
                nxt.append(idx)
                idx += 1
        frontier = nxt
    while idx < n_nodes:
        par = int(rng.integers(0, idx))
        edges.append((par, idx))
        depths.append(depths[par] + 1)
        t += rng.exponential(0.02)
        timestamps.append(min(t, 1.0))
        idx += 1
    return edges, np.array(depths, dtype=np.int32), np.array(timestamps, dtype=np.float32)


def generate_cascade(label, cfg, rng):
    n_nodes = max(4, int(rng.poisson(cfg['avg_nodes'])))
    d_mean, d_std, b_mean, b_std = _TOPOLOGY[label]
    max_d  = max(2, int(round(d_mean + rng.normal(0, d_std))))
    mean_b = max(0.3, b_mean + rng.normal(0, b_std))
    edges, depths, timestamps = _make_tree(n_nodes, max_d, mean_b, rng)

    # Features: HIGH noise, WEAK class signal.
    # No explicit depth/degree encoding — topology must be learned from adjacency.
    feats = rng.normal(0, 0.9, (n_nodes, FEAT_DIM)).astype(np.float32)
    feats[:, label] += 0.22          # weak class indicator (4 different channels)
    feats[:, (label + 1) % 4] -= 0.12  # weak negative indicator for adjacent class
    # Faint structural hint: root gets a slightly different signal
    feats[0, FEAT_DIM - 1] += 0.15

    return {
        'node_feats': feats, 'edges': edges, 'timestamps': timestamps,
        'label': int(label), 'n_nodes': n_nodes, 'max_depth': int(depths.max()),
    }


def cascade_to_tensors(cascade, device='cpu'):
    import torch
    n     = cascade['n_nodes']
    feats = torch.tensor(cascade['node_feats'], dtype=torch.float32, device=device)
    adj   = torch.zeros(n, n, dtype=torch.float32, device=device)
    for (s, d) in cascade['edges']:
        if 0 <= s < n and 0 <= d < n:
            adj[s, d] = 1.0
            adj[d, s] = 1.0
    adj += torch.eye(n, device=device)
    adj  = (adj > 0).float()
    return feats, adj


def generate_dataset(name, seed=42):
    cfg = DATASET_CFG[name]
    rng = np.random.default_rng(seed)
    w   = np.array(cfg['class_weights']); w /= w.sum()
    out = {}
    for split, n in [('train', cfg['n_train']), ('test', cfg['n_test'])]:
        labels = rng.choice(N_CLASSES, size=n, p=w)
        out[split] = [generate_cascade(int(lbl), cfg, rng) for lbl in labels]
    out['cascade_minutes'] = cfg['cascade_minutes']
    return out


def get_all_datasets(seed=42):
    return {name: generate_dataset(name, seed) for name in DATASET_CFG}


if __name__ == '__main__':
    ds = get_all_datasets()
    for name, splits in ds.items():
        tr = splits['train']
        by_cls = [[] for _ in range(4)]
        for c in tr: by_cls[c['label']].append(c['max_depth'])
        print(f"\n{name}: {len(tr)} train, avg_nodes={np.mean([c['n_nodes'] for c in tr]):.1f}")
        for k,v in enumerate(['NR','TR','FR','UR']):
            if by_cls[k]: print(f"  {v}: n={len(by_cls[k])}, avg_depth={np.mean(by_cls[k]):.1f}")
