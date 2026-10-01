import torch, torch.nn as nn, torch.nn.functional as F, math
EPS, BALL = 1e-7, 1e-3
VOCAB = 5000
SNAPS = [60, 240, 720, 2160]      # snapshot deadlines (minutes) for dynamic models

# ---------------- Poincare ball ops ----------------
def clamp(x, c):
    m = (1 - BALL) / c ** 0.5; n = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.where(n > m, x * m / n, x)
def exp0(v, c):
    sc = c ** 0.5; n = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return clamp(torch.tanh(sc * n) * v / (sc * n), c)
def log0(y, c):
    sc = c ** 0.5; n = torch.minimum(y.norm(dim=-1, keepdim=True).clamp(min=EPS), torch.as_tensor((1 - BALL) / sc))
    return torch.atanh(sc * n) / (sc * n) * y
def mob_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True); y2 = (y * y).sum(-1, keepdim=True); xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    return clamp(num / (1 + 2 * c * xy + c * c * x2 * y2).clamp(min=EPS), c)
def pdist(x, y, c):
    sc = c ** 0.5; n = torch.minimum(mob_add(-x, y, c).norm(dim=-1), torch.as_tensor((1 - BALL) / sc))
    return 2 / sc * torch.atanh(sc * n)

# ---------------- graph utils ----------------
def seg_softmax(e, idx, N):
    mx = torch.full((N,), -1e30).scatter_reduce(0, idx, e, 'amax', include_self=True)
    ex = (e - mx[idx]).exp(); s = torch.zeros(N).index_add_(0, idx, ex)
    return ex / s[idx].clamp(min=1e-16)
def pool_mean(h, batch, B):
    s = torch.zeros(B, h.size(-1)).index_add_(0, batch, h)
    c = torch.zeros(B).index_add_(0, batch, torch.ones_like(batch, dtype=torch.float)).clamp(min=1)
    return s / c[:, None]
def gcn_norm(src, dst, N, self_loops=True):
    if self_loops:
        ar = torch.arange(N); src = torch.cat([src, ar]); dst = torch.cat([dst, ar])
    deg = torch.zeros(N).index_add_(0, dst, torch.ones(len(dst)))
    w = deg[src].rsqrt() * deg[dst].rsqrt()
    return src, dst, w
def propagate(h, src, dst, w, N):
    return torch.zeros(N, h.size(-1)).index_add_(0, dst, h[src] * w[:, None])
def undirected(b):
    return torch.cat([b['src'], b['dst']]), torch.cat([b['dst'], b['src']])
def time_feat(t):  # arrival time in minutes -> scaled log feature
    return (torch.log1p(t / 60.0) / math.log1p(2160 / 60.0)).unsqueeze(-1)

class Encoder(nn.Module):   # sparse bag-of-words (TF counts, 5000 vocab) -> dense
    def __init__(self, d):
        super().__init__(); self.emb = nn.EmbeddingBag(VOCAB, d, mode='sum')
    def forward(self, b): return self.emb(b['words'], b['offs'])

# ---------------- D-HGE ----------------
class HypGAT(nn.Module):
    def __init__(self, d, c=1.0, curv=True, hyperbolic=True):
        super().__init__()
        self.W = nn.Linear(d, d); self.a_s = nn.Linear(d, 1, bias=False); self.a_d = nn.Linear(d, 1, bias=False)
        self.c, self.curv, self.hyp = c, curv, hyperbolic
    def forward(self, h, src, dst, N):
        c = self.c
        ht = self.W(log0(h, c)) if self.hyp else self.W(h)
        e = F.leaky_relu(self.a_s(ht)[src, 0] + self.a_d(ht)[dst, 0], 0.2)
        if self.curv and self.hyp:
            e = e - c * pdist(h[src], h[dst], c) ** 2
        a = seg_softmax(e, dst, N)
        agg = torch.zeros_like(ht).index_add_(0, dst, a[:, None] * ht[src])
        return exp0(agg + ht, c) if self.hyp else F.relu(agg + ht)

class DHGE(nn.Module):
    """Snapshot-recurrent hyperbolic model.  hyperbolic=False gives the Euclidean control;
    dynamic=False embeds only the latest snapshot (static variant)."""
    def __init__(self, d=64, c=1.0, curv=True, risk=True, hyperbolic=True, dynamic=True, dropout=0.3):
        super().__init__()
        self.c, self.risk, self.hyp, self.dyn = c, risk, hyperbolic, dynamic
        self.enc = Encoder(d); self.wt = nn.Linear(1, d, bias=False)
        self.ln = nn.LayerNorm(d); self.scale = nn.Parameter(torch.tensor(1.0 / d ** 0.5))
        self.layers = nn.ModuleList([HypGAT(d, c, curv, hyperbolic) for _ in range(2)])
        self.drop = nn.Dropout(dropout)
        din = d + (1 if risk else 0)
        self.gru = nn.GRUCell(din, d)
        self.cls = nn.Sequential(nn.Linear(d + din, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def embed(self, b):
        x = self.enc(b) + (self.wt(time_feat(b['t'])) if self.dyn else 0)
        x = self.scale * self.ln(x)                      # keep tangent inputs away from the ball boundary
        h = exp0(x, self.c) if self.hyp else x
        src, dst = undirected(b)
        for k, L in enumerate(self.layers):
            h = L(h, src, dst, b['N'])
            if k == 0:
                h = exp0(self.drop(log0(h, self.c)), self.c) if self.hyp else self.drop(h)
        g = pool_mean(log0(h, self.c) if self.hyp else h, b['batch'], b['B'])
        feats = [g]
        if self.risk:
            r = pool_mean(h.norm(dim=-1, keepdim=True), b['batch'], b['B'])  # cascade-risk score r_G
            feats.append(r)
        return torch.cat(feats, -1)
    def forward(self, snaps):              # snaps: list of batches, last = current observation
        if not self.dyn: snaps = snaps[-1:]
        s = None
        for b in snaps:
            z = self.embed(b); s = self.gru(z, s)
        return self.cls(torch.cat([s, z], -1))

# ---------------- HGCN (Chami et al. 2019), tangent-space aggregation at origin ----------------
class HGCN(nn.Module):
    def __init__(self, d=64, dropout=0.3):
        super().__init__()
        self.enc = Encoder(d); self.W = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(2)])
        self.kraw = nn.Parameter(torch.zeros(3))  # learnable curvatures (softplus)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def forward(self, snaps):
        b = snaps[-1]; cs = F.softplus(self.kraw) + 0.1
        h = exp0(self.enc(b), cs[0]); src, dst = undirected(b); s2, d2, w = gcn_norm(src, dst, b['N'])
        for l in range(2):
            c0, c1 = cs[l], cs[l + 1]
            mu = exp0(self.W[l](self.drop(log0(h, c0))), c0)                 # Mobius mat-vec
            mu = mob_add(mu, exp0(self.bias[l].expand_as(mu), c0), c0)       # hyperbolic bias
            agg = propagate(log0(mu, c0), s2, d2, w, b['N'])                 # tangent aggregation
            h = exp0(F.relu(agg), c1)                                         # activation, curvature change
        return self.cls(pool_mean(log0(h, cs[2]), b['batch'], b['B']))

# ---------------- BiGCN (Bian et al. 2020), original design ----------------
class BiGCNBranch(nn.Module):
    def __init__(self, d_h=64, d_o=64):
        super().__init__()
        self.c1 = Encoder(d_h); self.c2h = nn.Linear(d_h, d_o); self.c2r = Encoder(d_o)
    def forward(self, b, src, dst):
        N = b['N']; s, d, w = gcn_norm(src, dst, N)
        x1 = propagate(self.c1(b), s, d, w, N)                           # conv1 on raw TF features
        rootx = self.c2r.emb(b['words'], b['offs'])[b['root']][b['batch']]  # root raw feature enhancement (projected)
        h = F.dropout(F.relu(x1), 0.5, self.training)
        x2 = propagate(self.c2h(h) + rootx, s, d, w, N)                  # conv2 on [h ; x_root]
        x2 = F.relu(x2)
        out = torch.cat([x2, x1[b['root']][b['batch']]], -1)             # root hidden enhancement
        return pool_mean(out, b['batch'], b['B'])
class BiGCN(nn.Module):
    def __init__(self, d=64, dropedge=0.2):
        super().__init__(); self.td = BiGCNBranch(d, d); self.bu = BiGCNBranch(d, d); self.p = dropedge
        self.fc = nn.Linear(4 * d, 4)
    def forward(self, snaps):
        b = snaps[-1]; src, dst = b['src'], b['dst']
        if self.training and len(src):
            keep = torch.rand(len(src)) >= self.p; src, dst = src[keep], dst[keep]
        return self.fc(torch.cat([self.td(b, src, dst), self.bu(b, dst, src)], -1))

# ---------------- snapshot baselines ----------------
class GCN2(nn.Module):
    def __init__(self, d, dropout):
        super().__init__(); self.enc = Encoder(d); self.l2 = nn.Linear(d, d); self.drop = nn.Dropout(dropout)
    def forward(self, b):
        src, dst = undirected(b); s, dd, w = gcn_norm(src, dst, b['N'])
        h = self.drop(F.relu(propagate(self.enc(b), s, dd, w, b['N'])))
        h = F.relu(propagate(self.l2(h), s, dd, w, b['N']))
        return pool_mean(h, b['batch'], b['B'])
class DDGCN(nn.Module):   # dynamic propagation-graph branch of Sun et al. 2022 (knowledge branch omitted)
    def __init__(self, d=64, dropout=0.3):
        super().__init__(); self.g = GCN2(d, dropout); self.gru = nn.GRU(d, d, batch_first=True)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def forward(self, snaps):
        seq = torch.stack([self.g(b) for b in snaps], 1); _, hn = self.gru(seq)
        return self.cls(hn[0])
class DynGCN(nn.Module):  # Choi et al. 2021: per-snapshot GCN + temporal attention
    def __init__(self, d=64, dropout=0.3):
        super().__init__(); self.g = GCN2(d, dropout); self.att = nn.Linear(d, 1)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def forward(self, snaps):
        H = torch.stack([self.g(b) for b in snaps], 1); a = torch.softmax(self.att(H), 1)
        return self.cls((a * H).sum(1))
class CGNKP(nn.Module):   # continuous-time propagation with temporal encoding (knowledge module omitted)
    def __init__(self, d=64, dropout=0.3):
        super().__init__(); self.enc = Encoder(d); self.freq = nn.Parameter(torch.randn(d // 2))
        self.Ws = nn.Linear(2 * d, d); self.Wp = nn.Linear(d, d, bias=False); self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def forward(self, snaps):
        b = snaps[-1]; N = b['N']
        ang = torch.log1p(b['t'])[:, None] * self.freq[None]
        z = self.Ws(torch.cat([self.enc(b), torch.sin(ang), torch.cos(ang)], -1))
        parent = torch.full((N,), -1, dtype=torch.long); parent[b['dst']] = b['src']
        h = F.relu(z.clone()); D = int(b['depth'].max()) if N else 0
        for lev in range(1, D + 1):                                  # arrival-ordered recursion
            m = (b['depth'] == lev).nonzero(as_tuple=True)[0]
            h = h.index_copy(0, m, F.relu(z[m] + self.Wp(h[parent[m]])))
        return self.cls(pool_mean(self.drop(h), b['batch'], b['B']))

def build(name, dropout=0.3):
    p = dict(dropout=dropout)
    return {
     'D-HGE': lambda: DHGE(**p), 'HGCN': lambda: HGCN(**p), 'BiGCN': lambda: BiGCN(),
     'DDGCN': lambda: DDGCN(**p), 'DynGCN': lambda: DynGCN(**p), 'CGNKP': lambda: CGNKP(**p),
     'Euclidean': lambda: DHGE(hyperbolic=False, **p), 'Static': lambda: DHGE(dynamic=False, **p),
     'NoCurvAttn': lambda: DHGE(curv=False, **p), 'NoRiskScore': lambda: DHGE(risk=False, **p),
    }[name]()
