"""
D-HGE: Dynamic Hyperbolic Graph Embedding for misinformation cascade detection.
Poincaré ball operations and sparse curvature-aware graph attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

_EPS      = 1e-7
_BALL_EPS = 1e-3


def _clamp(x, c=1.0):
    max_norm = (1.0 - _BALL_EPS) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return torch.where(norm > max_norm, x * max_norm / norm, x)

def mob_add(x, y, c=1.0):
    x2  = (x * x).sum(-1, keepdim=True)
    y2  = (y * y).sum(-1, keepdim=True)
    xy  = (x * y).sum(-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    den = (1.0 + 2.0 * c * xy + c * c * x2 * y2).clamp(min=_EPS)
    return _clamp(num / den, c)

def p_dist(x, y, c=1.0):
    sc   = c ** 0.5
    diff = mob_add(-x, y, c)
    n    = diff.norm(dim=-1).clamp(max=1.0 - _BALL_EPS)
    return (2.0 / sc) * torch.atanh((sc * n + _EPS).clamp(max=1.0 - _EPS))

def exp0(v, c=1.0):
    sc = c ** 0.5
    nv = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return _clamp(torch.tanh(sc * nv) * v / (sc * nv), c)

def log0(y, c=1.0):
    sc = c ** 0.5
    ny = y.norm(dim=-1, keepdim=True).clamp(min=_EPS, max=1.0 - _BALL_EPS)
    return (torch.atanh(sc * ny) / (sc * ny)) * y


class HypGATLayer(nn.Module):
    """
    Curvature-aware graph attention (sparse edge-wise distance computation).
    Attention score for edge (i→j):
      e_ij = LeakyReLU(a_s^T·h_i + a_d^T·h_j) − c · d_P(h_i, h_j)²
    where d_P is the Poincaré geodesic distance.  Distances are computed
    only over actual edges (sparse), avoiding O(N²) dense computation.
    """
    def __init__(self, in_dim, out_dim, c=1.0, use_curv_penalty=True):
        super().__init__()
        self.c               = c
        self.use_curv_penalty = use_curv_penalty
        self.W   = nn.Linear(in_dim, out_dim)
        self.a_s = nn.Linear(out_dim, 1, bias=False)
        self.a_d = nn.Linear(out_dim, 1, bias=False)

    def forward(self, h_p, adj):
        c = self.c; N = h_p.size(0)
        h_t = self.W(log0(h_p, c))                # (N, out_dim) tangent

        e = self.a_s(h_t) + self.a_d(h_t).t()     # (N, N)
        e = F.leaky_relu(e, 0.2)

        if self.use_curv_penalty:
            rows, cols = adj.nonzero(as_tuple=True)
            if rows.numel() > 0:
                d2 = p_dist(h_p[rows], h_p[cols], c) ** 2  # sparse edges only
                penalty = torch.zeros(N, N, device=h_p.device)
                penalty[rows, cols] = d2
                e = e - c * penalty

        # Column-wise softmax: alpha[j,i] = weight of source i for dest j
        e_in  = e.t().masked_fill(adj.t() == 0, -1e9)
        alpha = torch.softmax(e_in, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        agg   = alpha @ h_t
        return _clamp(exp0(agg + h_t, c), c)


class EuclidGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim)
        self.a_s = nn.Linear(out_dim, 1, bias=False)
        self.a_d = nn.Linear(out_dim, 1, bias=False)

    def forward(self, h, adj):
        h_t  = F.relu(self.W(h))
        e    = self.a_s(h_t) + self.a_d(h_t).t()
        e    = F.leaky_relu(e, 0.2)
        e_in = e.t().masked_fill(adj.t() == 0, -1e9)
        alpha = torch.softmax(e_in, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return F.relu(alpha @ h_t + h_t)


class DHGE(nn.Module):
    """Dynamic Hyperbolic Graph Embedding (main model)."""
    def __init__(self, feat_dim=16, hidden=32, n_classes=4, c=1.0,
                 dropout=0.3, use_curv_penalty=True, use_risk_score=True,
                 n_layers=2):
        super().__init__()
        self.c              = c
        self.use_risk_score = use_risk_score
        self.feat_proj      = nn.Linear(feat_dim, hidden)
        self.layers         = nn.ModuleList([
            HypGATLayer(hidden, hidden, c, use_curv_penalty)
            for _ in range(n_layers)
        ])
        self.drop  = nn.Dropout(dropout)
        cls_in     = hidden + 1 if use_risk_score else hidden
        self.cls   = nn.Sequential(
            nn.Linear(cls_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, node_feats, adj, timestamps=None):
        c = self.c
        h = exp0(F.relu(self.feat_proj(node_feats)), c)
        for k, layer in enumerate(self.layers):
            h = layer(h, adj)
            if k < len(self.layers) - 1:
                h = exp0(self.drop(log0(h, c)), c)
        risk = h.norm(dim=-1).mean(dim=0, keepdim=True)   # cascade-risk score
        g    = log0(h, c).mean(dim=0)
        feat = torch.cat([g, risk], dim=0) if self.use_risk_score else g
        return self.cls(feat.unsqueeze(0)).squeeze(0)


class EuclideanGNN(nn.Module):
    """Euclidean ablation control: structurally identical to D-HGE without Poincaré ops."""
    def __init__(self, feat_dim=16, hidden=32, n_classes=4, dropout=0.3):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden)
        self.layer1    = EuclidGATLayer(hidden, hidden)
        self.layer2    = EuclidGATLayer(hidden, hidden)
        self.drop      = nn.Dropout(dropout)
        self.cls       = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )
    def forward(self, node_feats, adj, timestamps=None):
        h = F.relu(self.feat_proj(node_feats))
        h = self.layer1(h, adj); h = self.drop(h)
        h = self.layer2(h, adj)
        return self.cls(h.mean(dim=0).unsqueeze(0)).squeeze(0)
