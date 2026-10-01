"""Real Twitter15/16 cascades (Ma et al. 2017) with standard 5-fold splits and
arrival-time bins recovered from the time-truncated releases of STS-NN."""
import os, numpy as np, torch
ROOT = '/home/claude/STS-NN/dataset'
DEADLINES = [0, 60, 120, 240, 480, 720, 1440, 2160]
LAB = {'non-rumor': 0, 'news': 0, 'true': 1, 'false': 2, 'unverified': 3}  # NR,TR,FR,UR

def _read(path):
    d = {}
    for l in open(path, encoding='utf-8', errors='ignore'):
        a = l.rstrip('\r\n').split('\t')
        if len(a) < 6: a += [''] * (6 - len(a))
        d.setdefault(a[0], {})[int(a[1])] = a
    return d

def load_fold(ds, k):
    labels = {}
    for l in open(f'{ROOT}/{ds}/label.txt'):
        l = l.strip()
        if l: y, e = l.split(':'); labels[e] = LAB[y]
    full = _read(f'{ROOT}/{ds}/data.strnn.vol5000.f{k}.et2160.txt')
    first = {}
    for T in DEADLINES:
        for e, nodes in _read(f'{ROOT}/{ds}/data.strnn.vol5000.f{k}.et{T}.txt').items():
            for i in nodes: first.setdefault((e, i), T)
    graphs = {}
    for e, nodes in full.items():
        n = len(nodes); par = np.full(n, -1); words = []
        for i in range(n):
            a = nodes[i]
            if i > 0:
                ps = [int(p) for p in a[2].split() if p != 'None' and int(p) < i]
                par[i] = ps[0] if ps else 0      # posts without a listed parent attach to the source
            words.append([int(w) for w in a[5].split()] or [4999])
        t = np.array([first.get((e, i), 2160) for i in range(n)], dtype=np.float32)
        t = np.maximum.accumulate(t)  # arrival order is chronological
        graphs[e] = dict(eid=e, par=par, words=words, t=t, y=labels[e])
    rd = lambda s: [x.strip() for x in open(f'{ROOT}/nfold/RNN{s}Set_{ds}{k}_tree.txt') if x.strip()]
    tr = [e for e in rd('train') if e in graphs]; te = [e for e in rd('test') if e in graphs]
    return graphs, tr, te

def truncate(g, n_keep=None, tau=None):
    n = len(g['par'])
    if tau is not None: n_keep = max(1, int((g['t'] <= tau).sum()))
    n_keep = min(n, max(1, n_keep))
    return dict(eid=g['eid'], par=g['par'][:n_keep], words=g['words'][:n_keep],
                t=g['t'][:n_keep], y=g['y'])

def collate(gs):
    """Batch of trees -> flat tensors. Edges parent->child (top-down)."""
    words, offs, src, dst, batch, t, root, depth = [], [], [], [], [], [], [], []
    base = 0
    for b, g in enumerate(gs):
        n = len(g['par'])
        for i in range(n):
            offs.append(len(words)); words.extend(g['words'][i])
        d = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            p = g['par'][i]; src.append(base + p); dst.append(base + i); d[i] = d[p] + 1
        depth.append(d); batch += [b] * n; t.append(g['t']); root.append(base); base += n
    T = lambda x, dt=torch.long: torch.tensor(np.asarray(x), dtype=dt)
    return dict(words=T(words), offs=T(offs), src=T(src), dst=T(dst), batch=T(batch),
                t=torch.tensor(np.concatenate(t)), root=T(root), depth=T(np.concatenate(depth)),
                y=T([g['y'] for g in gs]), B=len(gs), N=base)
