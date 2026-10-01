import sys, json, time, math, random, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, '/home/claude/rev')
from data import load_fold, truncate, collate, DEADLINES
from models import build, SNAPS
torch.set_num_threads(1)
LR = {'BiGCN': 5e-4}
def snaps_of(gs, tau=None, n_frac=None):
    if n_frac is not None: gs = [truncate(g, n_keep=math.ceil(n_frac * len(g['par']))) for g in gs]
    if tau is not None: gs = [truncate(g, tau=tau) for g in gs]
    return [collate([truncate(g, tau=S) for g in gs]) for S in SNAPS]
def batches(gs, bs, **kw): return [snaps_of(gs[i:i + bs], **kw) for i in range(0, len(gs), bs)]
@torch.no_grad()
def predict(m, bl):
    m.eval(); return torch.cat([F.softmax(m(s), -1) for s in bl]).numpy()
def run(ds, fold, seed, name, G=None, max_ep=100, patience=10, bs=128, frac=1.0, eval_early=True, lr=None, dropout=0.3):
    if G is None: G = load_fold(ds, fold)
    graphs, tr, te = G
    rng = random.Random(seed); tr = tr[:]; rng.shuffle(tr)
    nv = len(tr) // 10; va, tr = tr[:nv], tr[nv:]
    if frac < 1: tr = tr[:int(len(tr) * frac)]
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(name, dropout); lr = lr or LR.get(name, 1e-3)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Gtr = [graphs[e] for e in tr]; Gva = [graphs[e] for e in va]; Gte = [graphs[e] for e in te]
    vb = batches(Gva, 256); yv = torch.tensor([g['y'] for g in Gva])
    best, state, bad, t0 = 1e9, None, 0, time.time()
    for ep in range(max_ep):
        m.train(); rng.shuffle(Gtr)
        for i in range(0, len(Gtr), bs):
            gs = Gtr[i:i + bs]; s = snaps_of(gs)
            loss = F.cross_entropy(m(s), s[-1]['y']); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        vl = F.nll_loss(torch.log(torch.tensor(predict(m, vb)) + 1e-9), yv).item()
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state)
    val_acc = float((predict(m, vb).argmax(1) == yv.numpy()).mean())
    y = np.array([g['y'] for g in Gte]); P = predict(m, batches(Gte, 256)); yh = P.argmax(1)
    r = dict(ds=ds, fold=fold, seed=seed, model=name, lr=lr, dropout=dropout, epochs=ep + 1, secs=time.time() - t0, frac=frac, val_acc=val_acc, val_loss=best,
             acc=float((yh == y).mean()), f1=float(f1_score(y, yh, average='macro')),
             auc=float(roc_auc_score((y != 0).astype(int), 1 - P[:, 0])),
             per_class_f1=f1_score(y, yh, average=None).tolist())
    r['ids'] = te; r['correct'] = (yh == y).astype(int).tolist()
    if eval_early:
        P20 = predict(m, batches(Gte, 256, n_frac=0.2)); r['eda20'] = float((P20.argmax(1) == y).mean()); r['correct20'] = (P20.argmax(1) == y).astype(int).tolist()
        lag = np.full(len(y), float(DEADLINES[-1])); done = np.zeros(len(y), bool); accs = {}
        for T in DEADLINES:
            PT = predict(m, batches(Gte, 256, tau=T)); ok = (PT.argmax(1) == y)
            accs[T] = float(ok.mean()); hit = ok & (PT.max(1) >= 0.65) & ~done
            lag[hit] = T; done |= hit
        r['acc_at'] = accs; r['mdl'] = float(lag.mean()); r['detected_frac'] = float(done.mean())
    return r
if __name__ == '__main__':
    ds, fold, seed, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    print(json.dumps(run(ds, fold, seed, name)))
