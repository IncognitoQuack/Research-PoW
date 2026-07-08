"""
models.py
=========
- CNNBranch:  1-D convolutional encoder over the current flow's feature vector
              (features treated as a 1-D "signal" of length F).
- RNNBranch:  Bidirectional LSTM encoder over the trailing window of flows.
- CNNRNNEnsemble: late-fusion of both branches through a small MLP head.
- CNNOnly / RNNOnly: single-model ablation baselines using the same branches.
"""
import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, n_features: int, channels=(32, 64, 128), kernel_size=3, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            pad = kernel_size // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, ceil_mode=True),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.out_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, F) -> (B, 1, F)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return x


class RNNBranch(nn.Module):
    def __init__(self, n_features: int, hidden_size=128, num_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (B, W, F)
        out, (h_n, _) = self.lstm(x)
        # concat last layer's forward/backward hidden states
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return h  # (B, out_dim)


class CNNRNNEnsemble(nn.Module):
    """Hybrid ensemble: CNN branch on the current flow + RNN branch on the
    trailing window, fused through a small MLP classification head."""

    def __init__(self, n_features: int, n_classes: int, cnn_cfg: dict,
                 rnn_cfg: dict, fusion_cfg: dict):
        super().__init__()
        self.cnn = CNNBranch(n_features, **cnn_cfg)
        self.rnn = RNNBranch(n_features, **rnn_cfg)
        fusion_in = self.cnn.out_dim + self.rnn.out_dim
        hidden = fusion_cfg.get("hidden_size", 128)
        dropout = fusion_cfg.get("dropout", 0.4)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, window, current):
        cnn_feat = self.cnn(current)
        rnn_feat = self.rnn(window)
        fused = torch.cat([cnn_feat, rnn_feat], dim=-1)
        return self.head(fused)


class CNNOnly(nn.Module):
    def __init__(self, n_features: int, n_classes: int, cnn_cfg: dict):
        super().__init__()
        self.cnn = CNNBranch(n_features, **cnn_cfg)
        self.head = nn.Linear(self.cnn.out_dim, n_classes)

    def forward(self, window, current):
        return self.head(self.cnn(current))


class RNNOnly(nn.Module):
    def __init__(self, n_features: int, n_classes: int, rnn_cfg: dict):
        super().__init__()
        self.rnn = RNNBranch(n_features, **rnn_cfg)
        self.head = nn.Linear(self.rnn.out_dim, n_classes)

    def forward(self, window, current):
        return self.head(self.rnn(window))


def build_model(model_name: str, n_features: int, n_classes: int, model_cfg: dict):
    if model_name == "ensemble":
        return CNNRNNEnsemble(n_features, n_classes, model_cfg["cnn"],
                               model_cfg["rnn"], model_cfg["fusion"])
    if model_name == "cnn_only":
        return CNNOnly(n_features, n_classes, model_cfg["cnn"])
    if model_name == "rnn_only":
        return RNNOnly(n_features, n_classes, model_cfg["rnn"])
    raise ValueError(f"Unknown model_name: {model_name}")
