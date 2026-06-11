"""
baselines.py
All four comparison methods and the centralised oracle implemented under
identical experimental conditions (same MLP, same data splits, same seeds).

  B1  FedAvg          — McMahan et al., AISTATS 2017
  B2  DP-FedAvg       — McMahan et al., ICLR 2018 (DP-SGD + FedAvg)
  B3  Local-EWC       — Kirkpatrick et al., PNAS 2017 (no federation)
  B4  DP-FedEwc       — Lyu et al., KBS 2024 (local FIM, no aggregation)
  Oracle Centralised-EWC — upper bound (all data pooled, no DP)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    make_model, compute_fisher, add_dp_noise_to_fisher,
    ewc_penalty, compute_epsilon, MLP
)

np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_tensors(client_data: dict) -> dict:
    """Convert {k: (np_X_tr, np_y_tr, np_X_te, np_y_te)} → same with tensors."""
    out = {}
    for k, (Xtr, ytr, Xte, yte) in client_data.items():
        out[k] = (torch.tensor(Xtr), torch.tensor(ytr),
                  torch.tensor(Xte), torch.tensor(yte))
    return out


def _fedavg_aggregate(local_states: list, local_n: list) -> dict:
    """Weighted average of state_dicts by local dataset sizes."""
    total_n = sum(local_n)
    agg = copy.deepcopy(local_states[0])
    for key in agg:
        agg[key] = sum(
            local_states[k][key] * (local_n[k] / total_n)
            for k in range(len(local_states))
        )
    return agg


def _local_train(local_model: MLP, X: torch.Tensor, y: torch.Tensor,
                 n_epochs: int, batch_size: int, lr: float,
                 anchor_params=None, fim_acc=None, ewc_lambda=0.0,
                 clip_norm=None, sigma_grad=None) -> int:
    """
    Standard (or DP) local SGD for n_epochs. Returns number of batches
    processed (for privacy accounting).
    """
    local_model.train()
    opt = torch.optim.Adam(local_model.parameters(), lr=lr)
    n = X.shape[0]
    n_steps = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            X_b, y_b = X[idx], y[idx]

            opt.zero_grad()
            logits = local_model(X_b).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y_b)

            if anchor_params is not None and fim_acc is not None:
                loss = loss + ewc_penalty(local_model, anchor_params,
                                          fim_acc, ewc_lambda)
            loss.backward()

            if clip_norm is not None:
                nn.utils.clip_grad_norm_(local_model.parameters(), clip_norm)
                if sigma_grad is not None:
                    for param in local_model.parameters():
                        if param.grad is not None:
                            noise = (torch.randn_like(param.grad)
                                     * sigma_grad * clip_norm / len(idx))
                            param.grad.data.add_(noise)
            opt.step()
            n_steps += 1

    return n_steps


# ---------------------------------------------------------------------------
# B1  FedAvg (no DP, no continual learning)
# ---------------------------------------------------------------------------

def run_fedavg(dataset: dict, config: dict) -> tuple:
    """
    Standard FedAvg.  No EWC, no DP.
    Returns (R_matrix, epsilon) where epsilon=inf (no privacy guarantee).
    """
    n_tasks = dataset['n_tasks']
    n_clients = dataset['n_clients']
    n_feat = dataset['n_feat']
    R = np.zeros((n_tasks, n_tasks))

    model = make_model(n_feat)
    tensors = {t: _to_tensors(dataset['tasks'][t]) for t in range(n_tasks)}

    for t in range(n_tasks):
        t_tensors = tensors[t]
        for rnd in range(config['n_rounds']):
            local_states, local_n = [], []
            for k in range(n_clients):
                X_k, y_k, _, _ = t_tensors[k]
                lm = copy.deepcopy(model)
                _local_train(lm, X_k, y_k, config['n_epochs'],
                             config['batch_size'], config['lr'])
                local_states.append(lm.state_dict())
                local_n.append(X_k.shape[0])
            model.load_state_dict(_fedavg_aggregate(local_states, local_n))

        # Evaluate on all tasks
        for j in range(n_tasks):
            R[t, j] = _eval_all_clients(model, tensors[j], n_clients)

    return R, float('inf')


# ---------------------------------------------------------------------------
# B2  DP-FedAvg (DP-SGD + FedAvg, no EWC)
# ---------------------------------------------------------------------------

def run_dp_fedavg(dataset: dict, config: dict) -> tuple:
    """
    FedAvg with DP-SGD local training.  No EWC continual learning.
    """
    n_tasks = dataset['n_tasks']
    n_clients = dataset['n_clients']
    n_feat = dataset['n_feat']
    R = np.zeros((n_tasks, n_tasks))

    sigma = config.get('sigma_grad', 1.5)
    clip  = config['clip_norm']

    model = make_model(n_feat)
    tensors = {t: _to_tensors(dataset['tasks'][t]) for t in range(n_tasks)}
    total_eps = 0.0

    for t in range(n_tasks):
        t_tensors = tensors[t]
        steps_this_task = 0
        avg_n = 0.0

        for rnd in range(config['n_rounds']):
            local_states, local_n = [], []
            for k in range(n_clients):
                X_k, y_k, _, _ = t_tensors[k]
                lm = copy.deepcopy(model)
                steps = _local_train(lm, X_k, y_k, config['n_epochs'],
                                     config['batch_size'], config['lr'],
                                     clip_norm=clip, sigma_grad=sigma)
                local_states.append(lm.state_dict())
                local_n.append(X_k.shape[0])
                steps_this_task += steps
                avg_n += X_k.shape[0]
            model.load_state_dict(_fedavg_aggregate(local_states, local_n))

        avg_n /= (config['n_rounds'] * n_clients)
        eps_t = compute_epsilon(sigma, int(avg_n), config['batch_size'],
                                steps_this_task // n_clients, delta=1e-5)
        total_eps += eps_t

        for j in range(n_tasks):
            R[t, j] = _eval_all_clients(model, tensors[j], n_clients)

    return R, total_eps


# ---------------------------------------------------------------------------
# B3  Local-EWC (no federation — each hospital trains independently)
# ---------------------------------------------------------------------------

def run_local_ewc(dataset: dict, config: dict) -> tuple:
    """
    Each client trains independently with EWC regularisation.
    No federation — models never leave each hospital.
    R[t, j] = mean AUC across clients evaluated on task j after training t tasks.
    """
    n_tasks = dataset['n_tasks']
    n_clients = dataset['n_clients']
    n_feat = dataset['n_feat']
    R = np.zeros((n_tasks, n_tasks))

    tensors = {t: _to_tensors(dataset['tasks'][t]) for t in range(n_tasks)}

    # One model per client
    client_models = [make_model(n_feat) for _ in range(n_clients)]
    anchors = [None] * n_clients
    fims    = [None] * n_clients

    for t in range(n_tasks):
        t_tensors = tensors[t]
        for k in range(n_clients):
            X_k, y_k, _, _ = t_tensors[k]
            lm = client_models[k]
            for _ in range(config['n_rounds']):
                _local_train(lm, X_k, y_k, config['n_epochs'],
                             config['batch_size'], config['lr'],
                             anchor_params=anchors[k], fim_acc=fims[k],
                             ewc_lambda=config['ewc_lambda'])
            # Update FIM and anchor
            fim_k = compute_fisher(lm, X_k, y_k)
            if fims[k] is None:
                fims[k] = {n: v.clone() for n, v in fim_k.items()}
            else:
                for nm in fims[k]:
                    fims[k][nm] = fims[k][nm] + fim_k[nm]
            anchors[k] = {nm: p.clone().detach()
                          for nm, p in lm.named_parameters()}

        # Evaluate: average AUC across clients
        for j in range(n_tasks):
            aucs = []
            for k in range(n_clients):
                _, _, X_te, y_te = tensors[j][k]
                aucs.append(_auc_single(client_models[k], X_te, y_te))
            R[t, j] = float(np.mean(aucs))

    return R, float('inf')


# ---------------------------------------------------------------------------
# B4  DP-FedEwc (federated + DP + LOCAL Fisher — no aggregation)
# ---------------------------------------------------------------------------

def run_dp_fedewc(dataset: dict, config: dict) -> tuple:
    """
    DP-FedEwc (Lyu et al., KBS 2024 spirit):
    Federation + DP-SGD + EWC with LOCALLY computed (and locally noise-added)
    Fisher.  Each client uses only its own FIM — no cross-client aggregation.
    This is the closest existing baseline to the proposed method.
    """
    n_tasks = dataset['n_tasks']
    n_clients = dataset['n_clients']
    n_feat = dataset['n_feat']
    R = np.zeros((n_tasks, n_tasks))

    sigma_grad = config.get('sigma_grad', 1.5)
    sigma_fim  = config.get('sigma_fim', 0.6)
    clip       = config['clip_norm']

    model = make_model(n_feat)
    tensors = {t: _to_tensors(dataset['tasks'][t]) for t in range(n_tasks)}

    # Each client maintains its own local accumulated FIM and anchor
    local_anchors = [None] * n_clients
    local_fims    = [None] * n_clients

    total_eps = 0.0

    for t in range(n_tasks):
        t_tensors = tensors[t]
        steps_this_task = 0
        avg_n = 0.0

        for rnd in range(config['n_rounds']):
            local_states, local_n = [], []
            for k in range(n_clients):
                X_k, y_k, _, _ = t_tensors[k]
                lm = copy.deepcopy(model)
                steps = _local_train(
                    lm, X_k, y_k, config['n_epochs'],
                    config['batch_size'], config['lr'],
                    anchor_params=local_anchors[k],
                    fim_acc=local_fims[k],
                    ewc_lambda=config['ewc_lambda'],
                    clip_norm=clip, sigma_grad=sigma_grad
                )
                local_states.append(lm.state_dict())
                local_n.append(X_k.shape[0])
                steps_this_task += steps
                avg_n += X_k.shape[0]
            model.load_state_dict(_fedavg_aggregate(local_states, local_n))

        # LOCAL Fisher update for each client (with DP noise, no aggregation)
        for k in range(n_clients):
            X_k, y_k, _, _ = t_tensors[k]
            fim_k = compute_fisher(model, X_k, y_k)
            # Each client adds its own noise — effective std remains sigma_fim
            noisy_fim_k = add_dp_noise_to_fisher(fim_k, sigma_fim=sigma_fim)
            if local_fims[k] is None:
                local_fims[k] = {nm: v.clone() for nm, v in noisy_fim_k.items()}
            else:
                for nm in local_fims[k]:
                    local_fims[k][nm] = local_fims[k][nm] + noisy_fim_k[nm]
            local_anchors[k] = {nm: p.clone().detach()
                                 for nm, p in model.named_parameters()}

        avg_n /= (config['n_rounds'] * n_clients)
        eps_t = compute_epsilon(sigma_grad, int(avg_n), config['batch_size'],
                                steps_this_task // n_clients, delta=1e-5)
        total_eps += eps_t

        for j in range(n_tasks):
            R[t, j] = _eval_all_clients(model, tensors[j], n_clients)

    return R, total_eps


# ---------------------------------------------------------------------------
# Oracle  Centralised-EWC (upper bound — all data pooled, no DP)
# ---------------------------------------------------------------------------

def run_centralised_ewc(dataset: dict, config: dict) -> tuple:
    """
    Upper-bound oracle: all client data pooled per task, standard EWC.
    Not a federated method; used only as a reference ceiling.
    """
    n_tasks = dataset['n_tasks']
    n_clients = dataset['n_clients']
    n_feat = dataset['n_feat']
    R = np.zeros((n_tasks, n_tasks))

    model = make_model(n_feat)
    tensors = {t: _to_tensors(dataset['tasks'][t]) for t in range(n_tasks)}

    anchor_params = None
    fim_acc       = None

    for t in range(n_tasks):
        # Pool all clients for this task
        X_all = torch.cat([tensors[t][k][0] for k in range(n_clients)])
        y_all = torch.cat([tensors[t][k][1] for k in range(n_clients)])

        _local_train(model, X_all, y_all,
                     config['n_epochs'] * config['n_rounds'],
                     config['batch_size'], config['lr'],
                     anchor_params=anchor_params,
                     fim_acc=fim_acc,
                     ewc_lambda=config['ewc_lambda'])

        fim_t = compute_fisher(model, X_all, y_all)
        if fim_acc is None:
            fim_acc = {nm: v.clone() for nm, v in fim_t.items()}
        else:
            for nm in fim_acc:
                fim_acc[nm] = fim_acc[nm] + fim_t[nm]
        anchor_params = {nm: p.clone().detach()
                         for nm, p in model.named_parameters()}

        for j in range(n_tasks):
            X_te = torch.cat([tensors[j][k][2] for k in range(n_clients)])
            y_te = torch.cat([tensors[j][k][3] for k in range(n_clients)])
            R[t, j] = _auc_single(model, X_te, y_te)

    return R, float('inf')


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

from sklearn.metrics import roc_auc_score


def _auc_single(model: nn.Module, X_te: torch.Tensor,
                y_te: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te).squeeze()).numpy()
    y_np = y_te.numpy()
    if len(np.unique(y_np)) < 2:
        return 0.5
    return float(roc_auc_score(y_np, probs))


def _eval_all_clients(model: nn.Module, t_tensors: dict,
                      n_clients: int) -> float:
    """Mean AUC across all clients for one task."""
    aucs = []
    for k in range(n_clients):
        _, _, X_te, y_te = t_tensors[k]
        aucs.append(_auc_single(model, X_te, y_te))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from data_generator import generate_mimic_iv_sim
    torch.manual_seed(42)
    np.random.seed(42)

    ds = generate_mimic_iv_sim(n_clients=5, n_train=80, n_test=30)
    cfg = {
        'n_rounds': 1, 'n_epochs': 1, 'batch_size': 32,
        'lr': 1e-3, 'clip_norm': 1.0, 'sigma_grad': 1.5,
        'sigma_fim': 0.6, 'ewc_lambda': 5.0
    }
    for name, fn in [('FedAvg', run_fedavg),
                     ('DP-FedAvg', run_dp_fedavg),
                     ('Local-EWC', run_local_ewc),
                     ('DP-FedEwc', run_dp_fedewc),
                     ('Centralised-EWC', run_centralised_ewc)]:
        R, eps = fn(ds, cfg)
        print(f"[{name}] R[0,0]={R[0,0]:.3f}, eps={eps:.2f}")
    print("baselines.py OK")
