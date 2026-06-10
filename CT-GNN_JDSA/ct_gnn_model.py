"""
CT-GNN: Causal Temporal Graph Neural Network
Architecture: TCE -> CTA -> CGAT -> Decoder -> Propagation Scorer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from typing import Optional


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
    def forward(self, x):
        return self.conv(tfunc.pad(x, (self.pad, 0)))


class TemporalConvEncoder(nn.Module):
    def __init__(self, in_feat, hidden, n_layers=3, kernel=3, dropout=0.1):
        super().__init__()
        self.layers  = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.drop    = nn.Dropout(dropout)
        ch = in_feat
        for i in range(n_layers):
            self.layers.append(_CausalConv1d(ch, hidden, kernel, 2**i))
            self.norms.append(nn.LayerNorm(hidden))
            ch = hidden
    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x = x.permute(0,2,3,1).reshape(B*N, C, T)
        for conv, norm in zip(self.layers, self.norms):
            h = tfunc.gelu(conv(x))
            h = norm(h.permute(0,2,1)).permute(0,2,1)
            x = (x + h) if h.shape[1] == x.shape[1] else h
            x = self.drop(x)
        return x.reshape(B, N, x.shape[1], T).permute(0,3,1,2)  # (B,T,N,hid)


class CausalTemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, causal_adj=None):
        B, T, N, D = H.shape
        H_mean = H.mean(1)       # (B, N, D)
        Q = self.W_q(H_mean).view(B,N,self.n_heads,self.d_k).transpose(1,2)
        K = self.W_k(H_mean).view(B,N,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_v(H_mean).view(B,N,self.n_heads,self.d_k).transpose(1,2)
        sc = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if causal_adj is not None:
            mask = (causal_adj + torch.eye(N, device=H.device)).clamp(0,1).bool()
            sc = sc.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.nan_to_num(tfunc.softmax(sc, -1), nan=0.0)
        out  = torch.matmul(self.drop(attn), V).transpose(1,2).reshape(B,N,D)
        return self.norm(H_mean + self.W_o(out))


class CausalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = out_dim // n_heads
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        # attention vector: (n_heads, 2*d_head)
        self.a    = nn.Parameter(torch.Tensor(n_heads, 2*self.d_head))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, H, causal_adj):
        B, N, _ = H.shape
        Wh = self.W(H).view(B, N, self.n_heads, self.d_head)  # (B,N,h,d)
        # Pairwise attention: e[b,i,j,h] = LeakyReLU(a_h . [Wh_i || Wh_j])
        # shape trick: (B,N,1,h,d) and (B,1,N,h,d)
        hi = Wh.unsqueeze(2).expand(-1,-1,N,-1,-1)   # (B,N,N,h,d)
        hj = Wh.unsqueeze(1).expand(-1,N,-1,-1,-1)   # (B,N,N,h,d)
        cat= torch.cat([hi,hj], dim=-1)               # (B,N,N,h,2d)
        # self.a: (h, 2d) -> broadcast over (B,N,N)
        e  = tfunc.leaky_relu((cat * self.a).sum(-1), 0.2)  # (B,N,N,h)
        e  = e.permute(0,3,1,2)                       # (B,h,N,N)
        mask = (causal_adj + torch.eye(N, device=H.device)).clamp(0,1).bool()
        e    = e.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha = torch.nan_to_num(tfunc.softmax(e, -1), nan=0.0)
        alpha = self.drop(alpha)                      # (B,h,N,N)
        # Wh: (B,N,h,d) -> (B,h,N,d)
        Wh_ = Wh.permute(0,2,1,3)
        out  = torch.matmul(alpha, Wh_)              # (B,h,N,d)
        out  = out.permute(0,2,1,3).reshape(B, N, -1)
        return self.norm(self.proj(H) + out)


class CTGNN(nn.Module):
    def __init__(self, n_nodes, in_feat=1, hidden=32, gat_dim=32,
                 n_tcn_layers=3, n_gat_layers=2, n_heads=4,
                 window=32, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.window  = window
        self.tce     = TemporalConvEncoder(in_feat, hidden, n_tcn_layers, 3, dropout)
        self.cta     = CausalTemporalAttention(hidden, n_heads, dropout)
        self.gat_layers = nn.ModuleList()
        in_d = hidden
        for _ in range(n_gat_layers):
            self.gat_layers.append(CausalGATLayer(in_d, gat_dim, n_heads, dropout))
            in_d = gat_dim
        self.decoder = nn.Sequential(
            nn.Linear(gat_dim, hidden), nn.GELU(), nn.Linear(hidden, window))
        self.prop_w  = nn.Parameter(torch.ones(1))

    def forward(self, x, causal_adj, return_scores=False):
        B, T, N = x.shape
        H    = self.tce(x.unsqueeze(-1))           # (B,T,N,hidden)
        Z    = self.cta(H, causal_adj)             # (B,N,hidden)
        for gat in self.gat_layers:
            Z = gat(Z, causal_adj)                 # (B,N,gat_dim)
        recon = self.decoder(Z).permute(0,2,1)     # (B,T,N)
        if not return_scores:
            return recon
        anom  = (x - recon).pow(2).mean(1)         # (B,N)
        prop  = anom + self.prop_w * torch.matmul(anom, causal_adj.t().float())
        return recon, anom, prop

def select_threshold_pot(scores, level=0.99):
    import torch
    return float(torch.quantile(scores, level))

def detect_anomalies(scores, threshold):
    return (scores.max(dim=-1).values > threshold).long()

