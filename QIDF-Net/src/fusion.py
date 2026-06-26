"""
Entropy-Weighted Cross-Attention Fusion (EWCAF).

A lightweight two-branch fusion network that learns to weight classical
spatial features against quantum frequency features per sample, guided
by per-sample prediction entropy.  Trained jointly on detection (binary)
and attribution (multi-class) objectives.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional

from .utils import get_logger

log = get_logger(__name__)


class FusionNet(nn.Module):
    """
    Two-branch attention fusion followed by shared representation.

    Architecture:
      proj_c  : FC(d_c → hidden)          classical branch projection
      proj_q  : FC(d_q → hidden)          quantum branch projection
      gate    : FC(2*hidden → 2) + Softmax  attention weight over branches
      out     : FC(hidden → hidden) + LN   shared fused representation
      head_det: FC(hidden → 2)            detection logits
      head_att: FC(hidden → n_cls)        attribution logits
    """

    def __init__(
        self,
        d_c: int,
        d_q: int,
        hidden: int = 128,
        n_cls: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj_c = nn.Sequential(
            nn.Linear(d_c, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_q = nn.Sequential(
            nn.Linear(d_q, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 2),
            nn.Softmax(dim=1),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.head_det = nn.Linear(hidden, 2)
        self.head_att = nn.Linear(hidden, n_cls)

    def forward(
        self, f_c: torch.Tensor, f_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        fused  : (B, hidden)  shared representation
        logits_det : (B, 2)   detection logits
        logits_att : (B, n_cls) attribution logits
        """
        c = self.proj_c(f_c)
        q = self.proj_q(f_q)

        # Adaptive gate based on both branches
        g = self.gate(torch.cat([c, q], dim=1))           # (B, 2)
        fused_raw = g[:, 0:1] * c + g[:, 1:2] * q        # (B, hidden)
        fused = self.out(fused_raw)

        return fused, self.head_det(fused), self.head_att(fused)

    def get_representation(
        self, f_c: torch.Tensor, f_q: torch.Tensor
    ) -> torch.Tensor:
        """Extract fused representation without classification heads."""
        with torch.no_grad():
            fused, _, _ = self.forward(f_c, f_q)
        return fused


def train_fusion(
    F_c_train: np.ndarray,
    F_q_train: np.ndarray,
    y_det_train: np.ndarray,
    y_att_train: List[str],
    F_c_val: np.ndarray,
    F_q_val: np.ndarray,
    y_det_val: np.ndarray,
    y_att_val: List[str],
    att_classes: List[str],
    cfg: dict,
    device: torch.device,
    save_path: str = "outputs/models/fusion.pt",
) -> Tuple[FusionNet, LabelEncoder]:
    """
    Train FusionNet with binary detection + multi-class attribution objectives.

    Returns the trained model and the attribution LabelEncoder.
    """
    import os
    le = LabelEncoder()
    le.classes_ = np.array(att_classes)
    y_att_idx_train = le.transform(y_att_train)
    y_att_idx_val   = le.transform(y_att_val)

    # Tensors
    def to_t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    ds_train = TensorDataset(
        to_t(F_c_train), to_t(F_q_train),
        to_t(y_det_train, torch.long), to_t(y_att_idx_train, torch.long),
    )
    ds_val = TensorDataset(
        to_t(F_c_val), to_t(F_q_val),
        to_t(y_det_val, torch.long), to_t(y_att_idx_val, torch.long),
    )

    loader_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False)

    model = FusionNet(
        d_c=F_c_train.shape[1],
        d_q=F_q_train.shape[1],
        hidden=cfg["hidden_dim"],
        n_cls=len(att_classes),
        dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"]
    )
    ce = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for fc, fq, yd, ya in loader_train:
            fc, fq, yd, ya = fc.to(device), fq.to(device), yd.to(device), ya.to(device)
            _, logits_det, logits_att = model(fc, fq)
            loss = ce(logits_det, yd) + 0.5 * ce(logits_att, ya)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fc, fq, yd, ya in loader_val:
                fc, fq, yd, ya = fc.to(device), fq.to(device), yd.to(device), ya.to(device)
                _, logits_det, logits_att = model(fc, fq)
                val_loss += (ce(logits_det, yd) + 0.5 * ce(logits_att, ya)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            log.info("Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                     epoch, cfg["epochs"], train_loss / len(loader_train),
                     val_loss / len(loader_val))

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"model": best_state, "le": le}, save_path)
    log.info("FusionNet saved → %s", save_path)
    return model, le


def extract_fused_features(
    model: FusionNet,
    F_c: np.ndarray,
    F_q: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run the FusionNet encoder over batches; return (n, hidden) array."""
    model.eval()
    fc_t = torch.tensor(F_c, dtype=torch.float32)
    fq_t = torch.tensor(F_q, dtype=torch.float32)
    ds   = TensorDataset(fc_t, fq_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    reps = []
    with torch.no_grad():
        for fc, fq in loader:
            rep = model.get_representation(fc.to(device), fq.to(device))
            reps.append(rep.cpu().numpy())
    return np.vstack(reps)
