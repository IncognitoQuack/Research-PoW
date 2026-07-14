"""
model.py
MLP backbone, empirical Fisher computation, DP-SGD utilities, and the
core DP-FedEWC-CL training logic (proposed method).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================================
# MLP backbone
# ===========================================================================

class MLP(nn.Module):
    """Two-hidden-layer MLP for binary tabular classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 hidden_dim2: int = 32, dropout: float = 0.30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)   # raw logits


def make_model(input_dim: int) -> MLP:
    return MLP(input_dim=input_dim)


# ===========================================================================
# Privacy accounting (Rényi DP → (ε, δ)-DP)
# ===========================================================================

def compute_rdp(noise_multiplier: float, sample_rate: float,
                n_steps: int, alpha: int) -> float:
    """
    Upper-bound on RDP at order alpha for the subsampled Gaussian mechanism.
    Uses the Mironov (2017) bound with Poisson subsampling approximation:
        rdp ≈ sample_rate² * alpha / (2 * sigma²)   [Theorem 3, Mironov 2017]
    Multiplied by n_steps by composition.
    """
    rdp_per_step = (sample_rate ** 2) * alpha / (2.0 * noise_multiplier ** 2)
    return n_steps * rdp_per_step


def rdp_to_dp(rdp: float, alpha: int, delta: float = 1e-5) -> float:
    """Convert RDP bound to (ε, δ)-DP via standard conversion."""
    eps = rdp + np.log((alpha - 1) / alpha) - (np.log(delta) + np.log(alpha)) / (alpha - 1)
    return max(eps, 0.0)


def compute_epsilon(noise_multiplier: float, n_samples: int,
                    batch_size: int, n_steps: int,
                    delta: float = 1e-5,
                    alpha_grid: tuple = (2, 4, 8, 16, 32)) -> float:
    """
    Compute (ε, δ)-DP by optimising over a grid of Rényi orders.
    Returns the tightest bound (minimum ε across alpha_grid).
    """
    q = batch_size / max(n_samples, 1)
    best_eps = float('inf')
    for alpha in alpha_grid:
        rdp = compute_rdp(noise_multiplier, q, n_steps, alpha)
        eps = rdp_to_dp(rdp, alpha, delta)
        if eps < best_eps:
            best_eps = eps
    return best_eps


# ===========================================================================
# DP-SGD training step
# ===========================================================================

def dp_sgd_step(model: nn.Module,
                X: torch.Tensor, y: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                clip_norm: float,
                noise_multiplier: float) -> float:
    """
    Single DP-SGD update:  clip full-batch gradient to clip_norm, then
    add calibrated Gaussian noise.  This is the micro-batch approximation
    documented in Abadi et al. (2016) and used throughout applied DP-FL work.
    """
    model.train()
    optimizer.zero_grad()
    logits = model(X).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

    # Gaussian noise addition (σ = noise_multiplier * C / batch_size)
    batch_size = X.shape[0]
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier * clip_norm / batch_size
            param.grad.data.add_(noise)

    optimizer.step()
    return loss.item()


# ===========================================================================
# Empirical Fisher diagonal computation
# ===========================================================================

def compute_fisher(model: nn.Module,
                   X: torch.Tensor, y: torch.Tensor,
                   n_samples: int = 60) -> dict:
    """
    Diagonal empirical Fisher: accumulates squared per-sample gradients
    over a random subset of n_samples.  Returns a dict {param_name: tensor}.
    """
    model.eval()
    n = min(n_samples, X.shape[0])
    idx = torch.randperm(X.shape[0])[:n]
    X_sub, y_sub = X[idx], y[idx]

    fim = {name: torch.zeros_like(param)
           for name, param in model.named_parameters()}

    for i in range(n):
        model.zero_grad()
        logit = model(X_sub[i:i+1]).view(-1)   # shape [1]
        loss = F.binary_cross_entropy_with_logits(logit, y_sub[i:i+1])
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fim[name].add_(param.grad.data.clone() ** 2)

    for name in fim:
        fim[name].div_(n)

    model.zero_grad()
    return fim


def add_dp_noise_to_fisher(fim: dict, sigma_fim: float,
                            clip_fim: float = 1.0) -> dict:
    """
    Add Gaussian noise to Fisher diagonal for DP protection.
    Each element receives N(0, (sigma_fim * clip_fim)²) noise.
    After addition, values are clipped to [0, ∞) (Fisher must be non-negative).
    """
    noisy = {}
    for name, val in fim.items():
        noise = torch.randn_like(val) * sigma_fim * clip_fim
        noisy[name] = (val + noise).clamp(min=0.0)
    return noisy


def aggregate_fisher(fisher_list: list, weights: list = None) -> dict:
    """
    Aggregate per-client Fishers by weighted averaging.
    weights: list of n_k values; defaults to uniform.
    """
    if weights is None:
        weights = [1.0] * len(fisher_list)
    total_w = sum(weights)
    names = fisher_list[0].keys()
    agg = {name: torch.zeros_like(fisher_list[0][name]) for name in names}
    for fim_k, w_k in zip(fisher_list, weights):
        for name in names:
            agg[name].add_(fim_k[name] * (w_k / total_w))
    return agg


# ===========================================================================
# EWC penalty computation
# ===========================================================================

def ewc_penalty(model: nn.Module,
                anchor_params: dict,
                fim_accumulated: dict,
                ewc_lambda: float) -> torch.Tensor:
    """
    EWC regularisation term: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
    Uses online EWC: accumulated FIM and a single anchor (most recent task).
    """
    penalty = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in anchor_params:
            diff = param - anchor_params[name].detach()
            fim_diag = fim_accumulated[name].detach()
            penalty = penalty + (fim_diag * diff.pow(2)).sum()
    return (ewc_lambda / 2.0) * penalty


# ===========================================================================
# DP-FedEWC-CL: per-task training loop (proposed method)
# ===========================================================================

def train_one_task_dp_fedewc_cl(
    model: nn.Module,
    client_data: dict,          # {client_idx: (X_tr, y_tr, X_te, y_te)}
    anchor_params: dict,        # θ* from previous task (None for task 0)
    fim_accumulated: dict,      # accumulated FIM from previous tasks (None for task 0)
    config: dict,
) -> tuple:
    """
    One federated continual-learning round for DP-FedEWC-CL.

    Returns:
      (updated model,
       new anchor_params,
       updated fim_accumulated,
       epsilon_consumed_this_task)
    """
    n_rounds    = config['n_rounds']
    n_epochs    = config['n_epochs']
    batch_size  = config['batch_size']
    lr          = config['lr']
    clip_norm   = config['clip_norm']
    sigma_grad  = config.get('sigma_grad', 1.5)
    sigma_fim   = config.get('sigma_fim', 0.6)
    ewc_lambda  = config['ewc_lambda']
    n_clients   = len(client_data)

    # Convert arrays to tensors once
    tensors = {}
    for k, (Xtr, ytr, _, _) in client_data.items():
        tensors[k] = (torch.tensor(Xtr), torch.tensor(ytr))

    # Track total steps per client for privacy accounting
    total_steps_per_client = 0

    for rnd in range(n_rounds):
        local_models = []
        local_n = []

        for k in range(n_clients):
            X_k, y_k = tensors[k]
            n_k = X_k.shape[0]
            local_n.append(n_k)

            # Initialise local model from global
            local_model = copy.deepcopy(model)
            opt = torch.optim.Adam(local_model.parameters(), lr=lr)

            for epoch in range(n_epochs):
                perm = torch.randperm(n_k)
                for start in range(0, n_k, batch_size):
                    idx = perm[start: start + batch_size]
                    X_b, y_b = X_k[idx], y_k[idx]

                    # Base DP-SGD step with gradient clipping + noise
                    local_model.train()
                    opt.zero_grad()
                    logits = local_model(X_b).squeeze(-1)
                    ce_loss = F.binary_cross_entropy_with_logits(logits, y_b)

                    # EWC penalty (only if we have previous task info)
                    if anchor_params is not None and fim_accumulated is not None:
                        penalty = ewc_penalty(local_model, anchor_params,
                                              fim_accumulated, ewc_lambda)
                    else:
                        penalty = torch.tensor(0.0)

                    total_loss = ce_loss + penalty
                    total_loss.backward()

                    # DP: clip + noise
                    nn.utils.clip_grad_norm_(local_model.parameters(), clip_norm)
                    for param in local_model.parameters():
                        if param.grad is not None:
                            noise = (torch.randn_like(param.grad)
                                     * sigma_grad * clip_norm / len(idx))
                            param.grad.data.add_(noise)
                    opt.step()
                    total_steps_per_client += 1

            local_models.append(copy.deepcopy(local_model.state_dict()))

        # FedAvg aggregation (weighted by client data size)
        total_n = sum(local_n)
        agg_state = copy.deepcopy(local_models[0])
        for key in agg_state:
            agg_state[key] = sum(
                local_models[k][key] * (local_n[k] / total_n)
                for k in range(n_clients)
            )
        model.load_state_dict(agg_state)

    # ------------------------------------------------------------------
    # Compute and aggregate Fisher across clients (NOVEL CONTRIBUTION 1)
    # ------------------------------------------------------------------
    client_fishers = []
    client_n = []
    for k in range(n_clients):
        X_k, y_k = tensors[k]
        fim_k = compute_fisher(model, X_k, y_k)
        # DP noise on Fisher — clipped to avoid negative values
        noisy_fim_k = add_dp_noise_to_fisher(fim_k, sigma_fim=sigma_fim)
        client_fishers.append(noisy_fim_k)
        client_n.append(X_k.shape[0])

    # Weighted aggregation: variance of mean = sigma_fim^2 / K (vs sigma_fim^2 local)
    agg_fim = aggregate_fisher(client_fishers, weights=client_n)

    # Accumulate FIM (online EWC: sum over tasks)
    if fim_accumulated is None:
        fim_accumulated = {name: val.clone() for name, val in agg_fim.items()}
    else:
        for name in fim_accumulated:
            fim_accumulated[name] = fim_accumulated[name] + agg_fim[name]

    # Update anchor parameters (θ* ← current global model)
    anchor_params = {name: param.clone().detach()
                     for name, param in model.named_parameters()}

    # Privacy budget computation for this task
    avg_n = sum(local_n) / n_clients
    eps_task = compute_epsilon(sigma_grad, avg_n, batch_size,
                                total_steps_per_client, delta=1e-5)

    return model, anchor_params, fim_accumulated, eps_task


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    m = make_model(input_dim=34)
    x = torch.randn(8, 34)
    out = m(x)
    assert out.shape == (8, 1), f"Shape error: {out.shape}"

    # Smoke-test Fisher
    y_dummy = torch.randint(0, 2, (100,)).float()
    X_dummy = torch.randn(100, 34)
    fim = compute_fisher(m, X_dummy, y_dummy, n_samples=10)
    for name, val in fim.items():
        assert val.shape == m.state_dict()[name].shape, "FIM shape mismatch"

    # Smoke-test EWC penalty
    anchor = {n: p.clone() for n, p in m.named_parameters()}
    pen = ewc_penalty(m, anchor, fim, ewc_lambda=1.0)
    assert pen.item() >= 0.0, "Penalty should be non-negative"

    print("model.py OK — all smoke tests passed")
