"""
Four Euclidean-space baselines for misinformation cascade detection.
All share the same cascade_to_tensors interface as DHGE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


def _norm_adj(adj):
    """Symmetric D^{-1/2} A D^{-1/2} normalisation."""
    deg       = adj.sum(-1).clamp(min=1e-9)
    d_inv_sqrt = deg ** -0.5
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)


class _GCNLayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W = nn.Linear(in_d, out_d)
    def forward(self, h, A):
        return F.relu(self.W(A @ h))


# ── BiGCN ─────────────────────────────────────────────────────────────────────
class BiGCN(nn.Module):
    """
    Bi-directional GCN on the static propagation tree.
    Reference: Bian et al., AAAI 2020.  DOI 10.1609/aaai.v34i01.5362
    """
    def __init__(self, feat_dim=16, hidden=32, n_classes=4, dropout=0.3):
        super().__init__()
        self.td  = nn.ModuleList([_GCNLayer(feat_dim, hidden), _GCNLayer(hidden, hidden)])
        self.bu  = nn.ModuleList([_GCNLayer(feat_dim, hidden), _GCNLayer(hidden, hidden)])
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, feats, adj, timestamps=None):
        A_td, A_bu = _norm_adj(adj), _norm_adj(adj.t())
        h_td, h_bu = feats, feats
        for gcn in self.td:
            h_td = self.drop(gcn(h_td, A_td))
        for gcn in self.bu:
            h_bu = self.drop(gcn(h_bu, A_bu))
        g = torch.cat([h_td.mean(0), h_bu.mean(0)])
        return self.cls(g.unsqueeze(0)).squeeze(0)


# ── DDGCN ─────────────────────────────────────────────────────────────────────
class DDGCN(nn.Module):
    """
    Dual Dynamic GCN: K temporal snapshot subgraphs fused with a GRU.
    Reference: Sun et al., AAAI 2022.  DOI 10.1609/aaai.v36i4.20385
    """
    K = 3

    def __init__(self, feat_dim=16, hidden=32, n_classes=4, dropout=0.3):
        super().__init__()
        self.gcn  = nn.ModuleList([_GCNLayer(feat_dim, hidden) for _ in range(self.K)])
        self.gru  = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, feats, adj, timestamps=None):
        n  = feats.size(0)
        ts = (timestamps if timestamps is not None
              else torch.linspace(0, 1, n, device=feats.device))
        thrs   = torch.linspace(0, 1, self.K + 1, device=feats.device)[1:]
        snaps  = []
        for k, thr in enumerate(thrs):
            mask    = (ts <= thr).float()
            adj_k   = adj * mask.unsqueeze(1) * mask.unsqueeze(0)
            adj_k   = _norm_adj(adj_k + 1e-9 * torch.eye(n, device=adj.device))
            snaps.append(self.drop(self.gcn[k](feats, adj_k)).mean(0, keepdim=True))
        seq     = torch.stack(snaps, dim=1)           # (1, K, hidden)
        _, h_n  = self.gru(seq)
        g       = h_n.squeeze()
        return self.cls(g.unsqueeze(0)).squeeze(0)


# ── DynGCN ────────────────────────────────────────────────────────────────────
class DynGCN(nn.Module):
    """
    Sequential snapshot GCN with attention-weighted temporal aggregation.
    Reference: Choi et al., PLOS ONE 2021.  DOI 10.1371/journal.pone.0256039
    """
    K = 5

    def __init__(self, feat_dim=16, hidden=32, n_classes=4, dropout=0.3):
        super().__init__()
        self.gcn   = nn.ModuleList([_GCNLayer(feat_dim, hidden) for _ in range(self.K)])
        self.attn  = nn.Linear(hidden, 1)
        self.drop  = nn.Dropout(dropout)
        self.cls   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, feats, adj, timestamps=None):
        n  = feats.size(0)
        ts = (timestamps if timestamps is not None
              else torch.linspace(0, 1, n, device=feats.device))
        thrs  = torch.linspace(0, 1, self.K + 1, device=feats.device)[1:]
        snaps = []
        for k, thr in enumerate(thrs):
            mask  = (ts <= thr).float()
            adj_k = adj * mask.unsqueeze(1) * mask.unsqueeze(0)
            adj_k = _norm_adj(adj_k + 1e-9 * torch.eye(n, device=adj.device))
            snaps.append(self.drop(self.gcn[k](feats, adj_k)).mean(0))
        snaps_t = torch.stack(snaps, dim=0)              # (K, hidden)
        alpha   = torch.softmax(self.attn(snaps_t), dim=0)
        g       = (alpha * snaps_t).sum(0)
        return self.cls(g.unsqueeze(0)).squeeze(0)


# ── CGNKP ─────────────────────────────────────────────────────────────────────
class CGNKP(nn.Module):
    """
    Continuous-time dynamic GNN with temporal positional encoding.
    Based on: Zhou & Gao, Mathematics 2024.  DOI 10.3390/math12223453
    """
    def __init__(self, feat_dim=16, hidden=32, n_classes=4, dropout=0.3):
        super().__init__()
        self.enc     = nn.Linear(feat_dim + 1, hidden)  # +1 temporal channel
        self.W_p     = nn.Linear(hidden, hidden)
        self.W_s     = nn.Linear(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.cls     = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, feats, adj, timestamps=None):
        n  = feats.size(0)
        ts = (timestamps if timestamps is not None
              else torch.linspace(0, 1, n, device=feats.device))
        t_col = ts.unsqueeze(1)
        h     = F.relu(self.enc(torch.cat([feats, t_col], dim=-1)))  # (N, hidden)

        order     = torch.argsort(ts)
        h_ord     = h[order]
        adj_ord   = adj[order][:, order]
        h_out     = torch.zeros_like(h_ord)

        for i in range(n):
            nbrs = (adj_ord[i] > 0).nonzero(as_tuple=True)[0]
            past = nbrs[nbrs < i]
            if past.numel() > 0:
                h_out[i] = F.relu(self.W_s(h_ord[i]) + self.W_p(h_out[past].mean(0)))
            else:
                h_out[i] = F.relu(self.W_s(h_ord[i]))

        g = self.drop(h_out).mean(0)
        return self.cls(g.unsqueeze(0)).squeeze(0)
