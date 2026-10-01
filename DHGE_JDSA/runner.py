"""Resumable job runner: python3 runner.py <budget_seconds>"""
import sys, json, time, os, collections; sys.path.insert(0,'/home/claude/rev')
from train import run; from data import load_fold
T0 = time.time(); BUDGET = float(sys.argv[1])
MAIN = ['D-HGE', 'HGCN', 'BiGCN', 'DDGCN', 'DynGCN', 'CGNKP']; ABL = ['Euclidean', 'Static', 'NoCurvAttn', 'NoRiskScore']
def rows(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def key(r, kind): return (kind, r['ds'], r['fold'], r['seed'] if kind == 'main' else 0, r['model'], r.get('lr'), r.get('dropout'))
done = {key(r, 'grid') for r in rows('grid2.jsonl')} | {key(r, 'main') for r in rows('results.jsonl')}
cache = {}
def G(ds, f):
    if (ds, f) not in cache: cache.clear(); cache[(ds, f)] = load_fold(ds, f)
    return cache[(ds, f)]
def budget_ok(): return time.time() - T0 < BUDGET
# 1) validation grid on fold 0 (seed 0) of each dataset
for ds in ['twitter16', 'twitter15']:
    for m in ['D-HGE', 'HGCN', 'DDGCN', 'DynGCN', 'CGNKP']:
        for lr in [1e-3, 5e-4]:
            for dp in [0.3, 0.5]:
                k = ('grid', ds, 0, 0, m, lr, dp)
                if k in done: continue
                if not budget_ok(): sys.exit(0)
                r = run(ds, 0, 0, m, G=G(ds, 0), eval_early=False, lr=lr, dropout=dp)
                open('grid2.jsonl', 'a').write(json.dumps({x: r[x] for x in ['ds','fold','seed','model','lr','dropout','val_loss','val_acc','epochs','secs']}) + '\n')
# 2) select by validation loss
best = {}
for r in rows('grid2.jsonl'):
    k = (r['ds'], r['model'])
    if k not in best or r['val_loss'] < best[k][0]: best[k] = (r['val_loss'], r['lr'], r['dropout'])
for ds in ['twitter15', 'twitter16']: best[(ds, 'BiGCN')] = (None, 5e-4, 0.5)
json.dump({f'{a}|{b}': v for (a, b), v in best.items()}, open('selected.json', 'w'), indent=1)
# 3) main (1 seed, 5 folds); ablations on Twitter16
for ds in ['twitter16', 'twitter15']:
    for fold in range(5):
        for m in MAIN + (ABL if ds == 'twitter16' else []):
            _, lr, dp = best[(ds, 'D-HGE' if m in ABL else m)]
            k = ('main', ds, fold, 0, m, lr, dp)
            if k in done: continue
            if not budget_ok(): sys.exit(0)
            r = run(ds, fold, 0, m, G=G(ds, fold), lr=lr, dropout=dp)
            open('results.jsonl', 'a').write(json.dumps(r) + '\n')
open('runner.done', 'w').write('ok')
