"""Baseline model implementations: MTAD-GAT, GANF (simplified), LSTM-VAE"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as tfunc


class MTADGATLayer(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.Tensor(n_heads, 2*self.d_head))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H):
        B, N, D = H.shape
        Wh = self.W(H).view(B, N, self.n_heads, self.d_head)
        hi = Wh.unsqueeze(2).expand(-1,-1,N,-1,-1)
        hj = Wh.unsqueeze(1).expand(-1,N,-1,-1,-1)
        cat = torch.cat([hi, hj], dim=-1)   # (B,N,N,h,2d)
        e   = tfunc.leaky_relu((cat * self.a).sum(-1), 0.2)  # (B,N,N,h)
        e   = e.permute(0,3,1,2)            # (B,h,N,N)
        alpha = tfunc.softmax(e, -1)
        Wh_  = Wh.permute(0,2,1,3)         # (B,h,N,d)
        out  = torch.matmul(alpha, Wh_).permute(0,2,1,3).reshape(B,N,D)
        return self.norm(H + out)


class MTADGAT(nn.Module):
    def __init__(self, n_nodes, window, hidden=32, n_gat=2,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.window  = window
        self.gru = nn.GRU(1, hidden, batch_first=True,
                          num_layers=2, dropout=dropout)
        self.gat_layers = nn.ModuleList(
            [MTADGATLayer(hidden, n_heads) for _ in range(n_gat)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, window))

    def forward(self, x, return_scores=False):
        B, T, N = x.shape
        x_flat  = x.permute(0,2,1).reshape(B*N, T, 1)
        _, h    = self.gru(x_flat)
        H       = h[-1].view(B, N, -1)
        for gat in self.gat_layers:
            H = gat(H)
        recon = self.decoder(H).permute(0,2,1)   # (B,T,N)
        if not return_scores:
            return recon
        anom = (x - recon).pow(2).mean(1)
        return recon, anom, anom


class GANFEncoder(nn.Module):
    def __init__(self, n_nodes, window, hidden=32, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.window  = window
        self.adj_raw = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.gru     = nn.GRU(1 + hidden, hidden, batch_first=True)
        self.graph_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden))
        self.mu_head = nn.Linear(hidden, 1)
        self.lv_head = nn.Linear(hidden, 1)

    def forward(self, x, return_scores=False):
        B, T, N = x.shape
        A = torch.sigmoid(self.adj_raw)
        A = A * (1 - torch.eye(N, device=x.device))
        h = torch.zeros(1, B*N, self.gru.hidden_size, device=x.device)
        mus, lvs = [], []
        for t in range(T):
            H = h[0].view(B, N, -1)
            msg = self.graph_mlp(torch.matmul(A, H)).view(B*N, -1)
            inp = torch.cat([x[:,t,:].reshape(B*N,1), msg], dim=-1).unsqueeze(1)
            _, h = self.gru(inp, h)
            H2  = h[0].view(B, N, -1)
            mus.append(self.mu_head(H2).squeeze(-1))
            lvs.append(self.lv_head(H2).squeeze(-1))
        mu = torch.stack(mus, 1)
        lv = torch.stack(lvs, 1)
        if not return_scores:
            return mu
        nll = 0.5*(lv + (x-mu).pow(2)/(lv.exp()+1e-6))
        anom = nll.mean(1)
        return mu, anom, anom


class LSTMVAE(nn.Module):
    def __init__(self, n_nodes, window, hidden=32, latent=16, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.window  = window
        self.latent  = latent
        self.enc_lstm = nn.LSTM(n_nodes, hidden, batch_first=True,
                                bidirectional=True, num_layers=2,
                                dropout=dropout)
        self.mu_fc  = nn.Linear(2*hidden, latent)
        self.lv_fc  = nn.Linear(2*hidden, latent)
        self.dec_init = nn.Linear(latent, hidden)
        self.dec_lstm = nn.LSTM(latent, hidden, batch_first=True,
                                num_layers=2, dropout=dropout)
        self.dec_fc = nn.Linear(hidden, n_nodes)

    def forward(self, x, return_scores=False):
        B, T, N = x.shape
        _, (h,_) = self.enc_lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        mu, lv = self.mu_fc(h_cat), self.lv_fc(h_cat)
        z = mu + (0.5*lv).exp() * torch.randn_like(mu)
        z_seq = z.unsqueeze(1).expand(-1,T,-1)
        h0 = self.dec_init(z).unsqueeze(0).repeat(2,1,1)
        dec_out,_ = self.dec_lstm(z_seq, (h0, torch.zeros_like(h0)))
        recon = self.dec_fc(dec_out)
        if not return_scores:
            return recon
        rec_err = (x - recon).pow(2)
        kl = -0.5*(1+lv-mu.pow(2)-lv.exp()).mean(-1,keepdim=True).unsqueeze(1)
        anom = rec_err.mean(1) + kl.squeeze(-1)
        return recon, anom, anom
