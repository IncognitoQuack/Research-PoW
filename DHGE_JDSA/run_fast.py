"""
Main experimental runner.  USAGE: python3 run_fast.py
Outputs: all_results.json  ablation.json
Runtime target: < 3 minutes on a single CPU core.
"""
import json, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(42); np.random.seed(42)

from data_generator import get_all_datasets, cascade_to_tensors
from model     import DHGE, EuclideanGNN
from baselines import BiGCN, DDGCN, DynGCN, CGNKP
from metrics   import compute_accuracy, compute_macro_f1, compute_auroc, compute_eda, compute_mdl

DEVICE    = 'cpu'
FEAT_DIM  = 16
N_CLS     = 4
HIDDEN    = 32
EPOCHS    = 12
EP_ABL    = 9
LR        = 1e-3
W_DECAY   = 1e-4
DROPOUT   = 0.3
CLIP      = 1.0


def _fwd(model, cas):
    feats, adj = cascade_to_tensors(cas, DEVICE)
    ts = torch.tensor(cas['timestamps'], dtype=torch.float32, device=DEVICE)
    try:
        return model(feats, adj, ts)
    except TypeError:
        return model(feats, adj)


def train_one(model, cascades, n_epochs=EPOCHS):
    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(n_epochs):
        for idx in np.random.permutation(len(cascades)):
            cas  = cascades[idx]
            lbl  = torch.tensor([cas['label']], dtype=torch.long)
            opt.zero_grad()
            logits = _fwd(model, cas)
            if logits is None or torch.isnan(logits).any():
                continue
            loss = crit(logits.unsqueeze(0), lbl)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
    return model


def evaluate(model, test_cascades, cascade_minutes):
    model.eval()
    logits_list, labels = [], []
    for cas in test_cascades:
        with torch.no_grad():
            logits = _fwd(model, cas)
        if logits is None or torch.isnan(logits).any():
            continue
        logits_list.append(logits.numpy())
        labels.append(cas['label'])

    if not logits_list:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'auroc': 0.5,
                'eda20': 0.0, 'mdl_minutes': cascade_minutes}

    arr  = np.array(logits_list)
    arr  = arr - arr.max(axis=1, keepdims=True)
    prob = np.exp(arr) / np.exp(arr).sum(axis=1, keepdims=True)
    pred = arr.argmax(axis=1)

    def mfn(f, a):
        with torch.no_grad():
            try:    return model(f, a)
            except: return model(f, a, None)

    eda20 = compute_eda(mfn, test_cascades, threshold=0.20)
    mdl   = compute_mdl(mfn, test_cascades, cascade_minutes)

    # median inference latency
    t0 = time.perf_counter()
    for cas in test_cascades[:8]:
        with torch.no_grad(): _fwd(model, cas)
    lat = (time.perf_counter() - t0) * 1000.0 / 8.0

    return {
        'accuracy':          round(float(compute_accuracy(labels, pred)), 4),
        'macro_f1':          round(float(compute_macro_f1(labels, pred, N_CLS)), 4),
        'auroc':             round(float(compute_auroc(labels, prob)), 4),
        'eda20':             round(float(eda20), 4),
        'mdl_minutes':       round(float(mdl), 2),
        'latency_median_ms': round(float(lat), 3),
    }


def run_ablation(train_data, test_data, cm):
    tr_sub = train_data[:130]   # subsample for speed
    te_sub = test_data[:45]
    variants = {
        'Full_DHGE':        DHGE(FEAT_DIM, HIDDEN, N_CLS, use_curv_penalty=True,  use_risk_score=True,  n_layers=2, dropout=DROPOUT),
        'Euclidean_GNN':    EuclideanGNN(FEAT_DIM, HIDDEN, N_CLS, dropout=DROPOUT),
        'Static_DHGE':      DHGE(FEAT_DIM, HIDDEN, N_CLS, use_curv_penalty=True,  use_risk_score=True,  n_layers=1, dropout=DROPOUT),
        'NoCurvAttn_DHGE':  DHGE(FEAT_DIM, HIDDEN, N_CLS, use_curv_penalty=False, use_risk_score=True,  n_layers=2, dropout=DROPOUT),
        'NoRiskScore_DHGE': DHGE(FEAT_DIM, HIDDEN, N_CLS, use_curv_penalty=True,  use_risk_score=False, n_layers=2, dropout=DROPOUT),
    }
    out = {}
    for name, mdl in variants.items():
        print(f"  [{name}]...", end=' ', flush=True)
        t0 = time.perf_counter()
        mdl = train_one(mdl, tr_sub, EP_ABL)
        m   = evaluate(mdl, te_sub, cm)
        print(f"acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} ({time.perf_counter()-t0:.1f}s)")
        out[name] = m
    return out


def main():
    t0_g = time.perf_counter()
    print("Generating synthetic datasets...")
    datasets = get_all_datasets(seed=42)

    all_results = {}
    for ds_name, splits in datasets.items():
        print(f"\n{'='*55}\nDataset: {ds_name}")
        tr, te, cm = splits['train'], splits['test'], splits['cascade_minutes']
        cfgs = {
            'D-HGE':  DHGE(FEAT_DIM, HIDDEN, N_CLS, dropout=DROPOUT),
            'BiGCN':  BiGCN(FEAT_DIM, HIDDEN, N_CLS, DROPOUT),
            'DDGCN':  DDGCN(FEAT_DIM, HIDDEN, N_CLS, DROPOUT),
            'DynGCN': DynGCN(FEAT_DIM, HIDDEN, N_CLS, DROPOUT),
            'CGNKP':  CGNKP(FEAT_DIM, HIDDEN, N_CLS, DROPOUT),
        }
        ds_res = {}
        for mname, mdl in cfgs.items():
            print(f"  {mname}...", end=' ', flush=True)
            t0 = time.perf_counter()
            mdl = train_one(mdl, tr)
            m   = evaluate(mdl, te, cm)
            print(f"done ({time.perf_counter()-t0:.1f}s) acc={m['accuracy']:.4f} "
                  f"f1={m['macro_f1']:.4f} auroc={m['auroc']:.4f} "
                  f"eda20={m['eda20']:.4f} mdl={m['mdl_minutes']:.1f}min")
            ds_res[mname] = m
        all_results[ds_name] = ds_res

    print(f"\n{'='*55}\nAblation on PHEME9:")
    ablation = run_ablation(datasets['PHEME9']['train'],
                             datasets['PHEME9']['test'],
                             datasets['PHEME9']['cascade_minutes'])

    with open('all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    with open('ablation.json', 'w') as f:
        json.dump(ablation, f, indent=2)

    print(f"\nSaved: all_results.json  ablation.json")
    print(f"Total wall time: {time.perf_counter()-t0_g:.1f} s")

if __name__ == '__main__':
    main()
