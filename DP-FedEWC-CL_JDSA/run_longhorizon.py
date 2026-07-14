"""
run_longhorizon.py
=========================================================================
Targeted experiment addressing the reviewer's specific criticism that
"the ablation study does not convincingly demonstrate the benefit of
budget recycling." Budget recycling reinvests residual privacy budget
from early-converging tasks into FIM-noise reduction for LATER tasks, so
its effect should compound as the task sequence lengthens; the original
4-task MIMIC-IV-Sim ablation gives it only 3 compounding steps.

This script repeats the full-vs-no-recycling component ablation on the
8-task MIMIC-IV-Sim-Long benchmark (data_generator_longhorizon.py) across
N_SEEDS independent replicates, and additionally reports BWT as a function
of task index (task 4-8, i.e. the second drift cycle) to show the gap
between the two variants widening over the sequence.

Output: longhorizon_results.json
"""

import argparse
import copy
import json
import time
import numpy as np
import torch
from scipy import stats

from data_generator_longhorizon import generate_mimic_iv_sim_long
from model import (
    make_model, compute_fisher, add_dp_noise_to_fisher, aggregate_fisher,
    compute_epsilon
)
from baselines import _to_tensors, _fedavg_aggregate, _local_train, _eval_all_clients
from metrics import compute_all_metrics, compute_bwt, compute_fwt, compute_avg_auc
from run_fast import BASE_CFG, N_CLIENTS


def _eval_ce_loss(model, tensors_dict, n_clients):
    """Mean BCE loss across clients — used only for the early-stopping
    criterion that lets task convergence speed (and hence privacy spend)
    vary genuinely across tasks in this long-horizon experiment."""
    import torch.nn.functional as F
    losses = []
    for k in range(n_clients):
        X_k, y_k = tensors_dict[k][0], tensors_dict[k][1]
        model.eval()
        with torch.no_grad():
            logits = model(X_k).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(logits, y_k).item())
    return float(np.mean(losses))


def run_dp_fedewc_cl_long(dataset, config, recycle=True, early_stop_tol=0.01,
                          min_rounds=2):
    """
    Same core algorithm as run_dp_fedewc_cl in run_fast.py, with one addition
    specific to this experiment: local training within a task stops early
    (after >= min_rounds) once the mean cross-entropy loss across clients
    stops improving by more than early_stop_tol between rounds. This lets
    "easy" tasks genuinely consume less of the round budget -- and hence
    less privacy budget -- than "hard" tasks, which is the regime budget
    recycling is designed to exploit. The MAIN comparison table
    (run_fast.py / run_multiseed.py) intentionally does NOT use early
    stopping, so as not to change the fixed-budget training protocol shared
    identically across all six compared methods; this script isolates the
    recycling mechanism on its own terms, over a longer task sequence.
    """
    n_tasks, n_clients, n_feat = dataset['n_tasks'], dataset['n_clients'], dataset['n_feat']
    R = np.zeros((n_tasks, n_tasks))
    model = make_model(n_feat)
    tensors = {t: _to_tensors(dataset['tasks'][t]) for t in range(n_tasks)}

    anchor_params = None
    fim_accumulated = None
    total_eps = 0.0
    eps_per_task = None
    sigma_fim_current = config.get('sigma_fim', 1.0)
    sigma_fim_trace = []

    for t in range(n_tasks):
        t_tensors = tensors[t]
        steps_this_task = 0
        local_n_list = []
        sigma_grad = config.get('sigma_grad', 1.3)
        prev_loss = None

        for rnd in range(config['n_rounds']):
            local_states, local_n = [], []
            for k in range(n_clients):
                X_k, y_k, _, _ = t_tensors[k]
                lm = copy.deepcopy(model)
                steps = _local_train(
                    lm, X_k, y_k, config['n_epochs'], config['batch_size'],
                    config['lr'], anchor_params=anchor_params,
                    fim_acc=fim_accumulated, ewc_lambda=config['ewc_lambda'],
                    clip_norm=config['clip_norm'], sigma_grad=sigma_grad,
                )
                local_states.append(lm.state_dict())
                local_n.append(X_k.shape[0])
                steps_this_task += steps
                if rnd == 0:
                    local_n_list.append(X_k.shape[0])
            model.load_state_dict(_fedavg_aggregate(local_states, local_n))

            # Early-stopping check (round-level convergence of this task)
            cur_loss = _eval_ce_loss(model, t_tensors, n_clients)
            if (rnd + 1) >= min_rounds and prev_loss is not None:
                if (prev_loss - cur_loss) < early_stop_tol:
                    prev_loss = cur_loss
                    break
            prev_loss = cur_loss

        client_fishers = []
        for k in range(n_clients):
            X_k, y_k, _, _ = t_tensors[k]
            fim_k = compute_fisher(model, X_k, y_k)
            noisy_fim_k = add_dp_noise_to_fisher(fim_k, sigma_fim=sigma_fim_current)
            client_fishers.append(noisy_fim_k)
        agg_fim = aggregate_fisher(client_fishers, weights=local_n_list)

        if fim_accumulated is None:
            fim_accumulated = {nm: v.clone() for nm, v in agg_fim.items()}
        else:
            for nm in fim_accumulated:
                fim_accumulated[nm] = fim_accumulated[nm] + agg_fim[nm]
        anchor_params = {nm: p.clone().detach() for nm, p in model.named_parameters()}

        avg_n = float(np.mean(local_n_list))
        eps_t = compute_epsilon(sigma_grad, int(avg_n), config['batch_size'],
                                steps_this_task // n_clients, delta=1e-5)
        total_eps += eps_t

        if recycle:
            if t == 0:
                eps_per_task = eps_t
            else:
                surplus = eps_per_task - eps_t
                if surplus > 0:
                    reduction = 0.10 * (surplus / max(eps_per_task, 1e-6))
                    sigma_fim_current = max(
                        sigma_fim_current * (1.0 - reduction),
                        sigma_fim_current * 0.70
                    )
        sigma_fim_trace.append(sigma_fim_current)

        for j in range(n_tasks):
            R[t, j] = _eval_all_clients(model, tensors[j], n_clients)

    return R, total_eps, sigma_fim_trace


def bwt_by_window(R, window):
    """BWT computed using only the last `window` tasks as the 'earlier' set,
    i.e. how much the model forgets tasks (T-window .. T-2) by task T-1.
    Lets us see whether the recycling gap widens over the second half of a
    long sequence."""
    T = R.shape[0]
    j_start = max(0, T - 1 - window)
    diffs = [R[T - 1, j] - R[j, j] for j in range(j_start, T - 1)]
    return float(np.mean(diffs)) if diffs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--n_tasks', type=int, default=8)
    args = parser.parse_args()

    results = {'full_recycling': {'avg_auc': [], 'bwt': [], 'bwt_2nd_half': [], 'fwt': []},
              'no_recycling':   {'avg_auc': [], 'bwt': [], 'bwt_2nd_half': [], 'fwt': []}}

    t0 = time.perf_counter()
    for s in range(args.seeds):
        data_seed = 42 * 1000 + s
        ds = generate_mimic_iv_sim_long(n_clients=N_CLIENTS, seed=data_seed,
                                        n_tasks=args.n_tasks)
        for variant, recycle in [('full_recycling', True), ('no_recycling', False)]:
            torch.manual_seed(2000 + s)
            R, eps, sigma_trace = run_dp_fedewc_cl_long(ds, BASE_CFG, recycle=recycle)
            results[variant]['avg_auc'].append(compute_avg_auc(R))
            results[variant]['bwt'].append(compute_bwt(R))
            results[variant]['bwt_2nd_half'].append(
                bwt_by_window(R, window=args.n_tasks // 2))
            results[variant]['fwt'].append(compute_fwt(R))
        print(f"  seed {s+1}/{args.seeds} done "
              f"(full BWT={results['full_recycling']['bwt'][-1]:+.4f}, "
              f"no-recycling BWT={results['no_recycling']['bwt'][-1]:+.4f})",
              flush=True)

    def summ(v):
        arr = np.asarray(v)
        return {'mean': round(float(arr.mean()), 4),
                'std': round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0}

    summary = {variant: {k: summ(v) for k, v in metrics.items()}
              for variant, metrics in results.items()}

    # Significance tests (paired across seeds)
    sig = {}
    for metric in ['bwt', 'bwt_2nd_half', 'avg_auc', 'fwt']:
        a = results['full_recycling'][metric]
        b = results['no_recycling'][metric]
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        t_stat, t_p = stats.ttest_rel(a, b)
        sig[metric] = {'wilcoxon_p': float(w_p), 'ttest_p': float(t_p),
                       'mean_diff': round(float(np.mean(a) - np.mean(b)), 4)}

    out = {'n_seeds': args.seeds, 'n_tasks': args.n_tasks,
          'summary': summary, 'significance': sig, 'raw': results}
    with open('longhorizon_results.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s")
    print(json.dumps({'summary': summary, 'significance': sig}, indent=2))


if __name__ == '__main__':
    main()
