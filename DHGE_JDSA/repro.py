"""BiGCN under its authors' protocol (random stratified 5-fold on the BiGCN data release,
early stopping on the test fold) versus the same folds with a held-out validation split."""
import sys, json, random, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/claude/rev')
from data import collate; from models import build
from sklearn.metrics import f1_score
torch.set_num_threads(1)
LAB = {'non-rumor': 0, 'news': 0, 'true': 1, 'false': 2, 'unverified': 3}
def load(ds):
    D = f'/home/claude/BiGCN/data/{ds}'; lab = {}
    for l in open(f'{D}/{ds}_label_All.txt'):
        a = l.rstrip().split('\t'); lab[a[2]] = LAB[a[0].lower()]
    raw = {}
    for l in open(f'{D}/data.TD_RvNN.vol_5000.txt'):
        a = l.rstrip('\n').split('\t'); raw.setdefault(a[0], []).append(a)
    G = {}
    for e, rows in raw.items():
        if e not in lab: continue
        byid = {int(a[2]): a for a in rows}
        kids = {}
        root = None
        for a in rows:
            if a[1] == 'None' or int(a[1]) not in byid: root = int(a[2]) if root is None else root
            else: kids.setdefault(int(a[1]), []).append(int(a[2]))
        order = [root]; seen = {root}
        for u in order:
            for v in sorted(kids.get(u, [])):
                if v not in seen: seen.add(v); order.append(v)
        order += [i for i in sorted(byid) if i not in seen]          # unreachable rows attach to root
        pos = {u: i for i, u in enumerate(order)}
        par = np.full(len(order), -1); words = []
        for i, u in enumerate(order):
            a = byid[u]
            if i > 0:
                p = int(a[1]) if a[1] != 'None' else -1
                par[i] = pos[p] if p in pos and pos[p] < i else 0
            w = []
            for kv in a[5].split():
                k, c = kv.split(':'); w += [int(k)] * int(c)
            words.append(w or [3999])
        G[e] = dict(eid=e, par=par, words=words, t=np.zeros(len(order), np.float32), y=lab[e])
    return G
def folds(G, seed):
    rng = random.Random(seed); by = {c: [e for e in G if G[e]['y'] == c] for c in range(4)}
    for c in by: rng.shuffle(by[c])
    out = []
    for k in range(5):
        te = []; tr = []
        for c, L in by.items():
            n = int(len(L) * 0.2); te += L[k*n:(k+1)*n]; tr += L[:k*n] + L[(k+1)*n:]
        out.append((tr, te))
    return out
def train_eval(G, tr, te, protocol, seed):
    torch.manual_seed(seed); rng = random.Random(seed); tr = tr[:]; rng.shuffle(tr)
    if protocol == 'val': nv = len(tr) // 10; sel, tr = tr[:nv], tr[nv:]
    else: sel = te                                   # original code: stopping monitored on the test fold
    m = build('BiGCN'); opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-4)
    Gtr = [G[e] for e in tr]; Gs = [G[e] for e in sel]; Gte = [G[e] for e in te]
    bs_sel = [collate(Gs[i:i+256]) for i in range(0, len(Gs), 256)]
    bs_te = [collate(Gte[i:i+256]) for i in range(0, len(Gte), 256)]
    def pred(bl):
        m.eval()
        with torch.no_grad(): return torch.cat([F.softmax(m([b]), -1) for b in bl])
    ys = torch.tensor([g['y'] for g in Gs]); best, bad, state = 1e9, 0, None
    for ep in range(200):
        m.train(); rng.shuffle(Gtr)
        for i in range(0, len(Gtr), 128):
            b = collate(Gtr[i:i+128]); loss = F.cross_entropy(m([b]), b['y'])
            opt.zero_grad(); loss.backward(); opt.step()
        vl = F.nll_loss(torch.log(pred(bs_sel) + 1e-9), ys).item()
        if vl < best: best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    m.load_state_dict(state); P = pred(bs_te).numpy(); y = np.array([g['y'] for g in Gte])
    return float((P.argmax(1) == y).mean()), float(f1_score(y, P.argmax(1), average='macro'))
if __name__ == '__main__':
    import os
    done = {(r['ds'], r['fold'], r['protocol']) for r in (json.loads(l) for l in open('/home/claude/rev/repro.jsonl'))} if os.path.exists('/home/claude/rev/repro.jsonl') else set()
    out = open('/home/claude/rev/repro.jsonl', 'a')
    for ds in ['Twitter16', 'Twitter15']:
        G = load(ds)
        for k, (tr, te) in enumerate(folds(G, 0)):
            for protocol in ['test', 'val']:
                if (ds, k, protocol) in done: continue
                acc, f1 = train_eval(G, tr, te, protocol, 0)
                out.write(json.dumps(dict(ds=ds, fold=k, protocol=protocol, acc=acc, f1=f1)) + '\n'); out.flush()
