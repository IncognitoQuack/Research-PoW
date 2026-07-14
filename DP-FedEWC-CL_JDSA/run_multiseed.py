"""
run_multiseed.py
=========================================================================
Statistically-rigorous replication protocol, added in response to
reviewer feedback that the original single-seed evaluation lacked
multiple independent runs, standard deviations, confidence intervals,
and significance testing.

For each of N_SEEDS independent replicates, a FRESH dataset is drawn
(different data-generation seed -> different missingness pattern, label
noise, client heterogeneity, coding noise, and train/test split) AND a
fresh model initialisation / training stochasticity is used. All six
methods are evaluated on every replicate under IDENTICAL data and
identical hyperparameters.

Outputs:
    multiseed_results.json   — per-method, per-dataset mean/std/CI over
                                N_SEEDS replicates, plus paired
                                Wilcoxon signed-rank tests of
                                DP-FedEWC-CL vs. the strongest DP
                                baseline (DP-FedEwc) for avg_auc, BWT,
                                and FWT.
    multiseed_ablation.json  — component ablation (full / no-recycling /
                                no-FIM-aggregation / no-EWC) repeated
                                over N_SEEDS replicates on MIMIC-IV-Sim,
                                with the same paired significance tests.

Run:
    python3 run_multiseed.py               # default 10 seeds
    python3 run_multiseed.py --seeds 20    # override
"""

import argparse
import copy
import json
import time
import numpy as np
import torch
from scipy import stats

from data_generator import (
    generate_mimic_iv_sim, generate_eicu_sim, generate_hirid_sim
)
from model import (
    make_model, compute_fisher, add_dp_noise_to_fisher,
    aggregate_fisher, compute_epsilon
)
from baselines import (
    run_fedavg, run_dp_fedavg, run_local_ewc,
    run_dp_fedewc, run_centralised_ewc,
    _to_tensors, _fedavg_aggregate, _local_train, _eval_all_clients
)
from metrics import compute_all_metrics
from run_fast import run_dp_fedewc_cl, BASE_CFG, N_CLIENTS

DATA_SEED_BASE = {
    'mimic_iv_sim': 42,
    'eicu_sim':     43,
    'hirid_sim':    44,
}
DATA_GENERATORS = {
    'mimic_iv_sim': generate_mimic_iv_sim,
    'eicu_sim':     generate_eicu_sim,
    'hirid_sim':    generate_hirid_sim,
}
METHODS = {
    'FedAvg':          run_fedavg,
    'DP_FedAvg':       run_dp_fedavg,
    'Local_EWC':       run_local_ewc,
    'DP_FedEwc':       run_dp_fedewc,
    'DP_FedEWC_CL':    run_dp_fedewc_cl,
    'Centralised_EWC': run_centralised_ewc,
}


# ---------------------------------------------------------------------------
# Ablation variants (component ablation only — repeated over seeds)
# ---------------------------------------------------------------------------

def _run_no_recycling(ds, cfg):
    """DP-FedEWC-CL with FIM aggregation but sigma_fim held fixed (no recycling)."""
    n_tasks, n_clients, n_feat = ds['n_tasks'], ds['n_clients'], ds['n_feat']
    R = np.zeros((n_tasks, n_tasks))
    model = make_model(n_feat)
    tensors = {t: _to_tensors(ds['tasks'][t]) for t in range(n_tasks)}
    anchor_params = None; fim_acc = None; total_eps = 0.0
    sigma_grad = cfg.get('sigma_grad', 1.5)
    sigma_fim  = cfg.get('sigma_fim', 1.0)   # NEVER adjusted (ablated component)
    for t in range(n_tasks):
        t_tensors = tensors[t]
        steps = 0; local_n_list = []
        for rnd in range(cfg['n_rounds']):
            local_states, local_n = [], []
            for k in range(n_clients):
                X_k, y_k, _, _ = t_tensors[k]
                lm = copy.deepcopy(model)
                s = _local_train(lm, X_k, y_k, cfg['n_epochs'],
                                 cfg['batch_size'], cfg['lr'],
                                 anchor_params=anchor_params, fim_acc=fim_acc,
                                 ewc_lambda=cfg['ewc_lambda'],
                                 clip_norm=cfg['clip_norm'], sigma_grad=sigma_grad)
                local_states.append(lm.state_dict())
                local_n.append(X_k.shape[0])
                steps += s
                if rnd == 0: local_n_list.append(X_k.shape[0])
            model.load_state_dict(_fedavg_aggregate(local_states, local_n))
        cfms = [add_dp_noise_to_fisher(compute_fisher(model, t_tensors[k][0], t_tensors[k][1]),
                sigma_fim=sigma_fim) for k in range(n_clients)]
        agg = aggregate_fisher(cfms, weights=local_n_list)
        if fim_acc is None: fim_acc = {nm: v.clone() for nm, v in agg.items()}
        else:
            for nm in fim_acc: fim_acc[nm] = fim_acc[nm] + agg[nm]
        anchor_params = {nm: p.clone().detach() for nm, p in model.named_parameters()}
        avg_n = float(np.mean(local_n_list))
        total_eps += compute_epsilon(sigma_grad, int(avg_n), cfg['batch_size'],
                                     steps // n_clients, delta=1e-5)
        for j in range(n_tasks):
            R[t, j] = _eval_all_clients(model, tensors[j], n_clients)
    return R, total_eps


def _run_no_fim_agg(ds, cfg):
    """DP-FedEWC-CL without FIM aggregation (each client keeps local FIM)."""
    return run_dp_fedewc(ds, cfg)


def _run_no_ewc(ds, cfg):
    """DP-FedEWC-CL without EWC (lambda=0 -> collapses to DP-FedAvg)."""
    return run_dp_fedavg(ds, cfg)


COMPONENT_VARIANTS = {
    'full_dp_fedewc_cl':  run_dp_fedewc_cl,
    'no_recycling':       _run_no_recycling,
    'no_fim_aggregation': _run_no_fim_agg,
    'no_ewc':             _run_no_ewc,
}


# ---------------------------------------------------------------------------
# Aggregation / statistics helpers
# ---------------------------------------------------------------------------

def _summ(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    mean = float(np.mean(finite)) if n else float('nan')
    std  = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / np.sqrt(n)
        ci95 = 1.96 * sem
    else:
        ci95 = 0.0
    return {'mean': round(mean, 4), 'std': round(std, 4),
            'ci95': round(ci95, 4), 'n': n, 'values': [round(v, 4) for v in finite]}


def _wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test; returns (statistic, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    if np.allclose(diffs, 0.0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return float('nan'), float('nan')


def _paired_ttest(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = stats.ttest_rel(a, b)
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Main comparison, repeated over seeds
# ---------------------------------------------------------------------------

def run_main_comparison(n_seeds, verbose=True):
    per_ds_method_metric = {}   # ds -> method -> metric -> [values across seeds]

    for ds_name, gen_fn in DATA_GENERATORS.items():
        per_ds_method_metric[ds_name] = {m: {'avg_auc': [], 'bwt': [], 'fwt': [],
                                              'privacy_epsilon': []}
                                          for m in METHODS}
        base_seed = DATA_SEED_BASE[ds_name]
        for s in range(n_seeds):
            data_seed = base_seed * 1000 + s
            torch.manual_seed(1000 + s)
            np.random.seed(1000 + s)
            ds = gen_fn(n_clients=N_CLIENTS, seed=data_seed)
            for method_name, method_fn in METHODS.items():
                torch.manual_seed(2000 + s)   # fresh init/training stochasticity per seed
                R, eps = method_fn(ds, BASE_CFG)
                m = compute_all_metrics(R, eps)
                per_ds_method_metric[ds_name][method_name]['avg_auc'].append(m['avg_auc'])
                per_ds_method_metric[ds_name][method_name]['bwt'].append(m['bwt'])
                per_ds_method_metric[ds_name][method_name]['fwt'].append(m['fwt'])
                per_ds_method_metric[ds_name][method_name]['privacy_epsilon'].append(
                    m['privacy_epsilon'])
            if verbose:
                print(f"  [{ds_name}] seed {s+1}/{n_seeds} done", flush=True)

    # Summarise
    summary = {}
    sig_tests = {}
    for ds_name, methods_dict in per_ds_method_metric.items():
        summary[ds_name] = {}
        for method_name, metric_dict in methods_dict.items():
            summary[ds_name][method_name] = {
                k: _summ(v) for k, v in metric_dict.items()
            }
        # Paired significance: DP_FedEWC_CL vs DP_FedEwc (closest DP baseline)
        sig_tests[ds_name] = {}
        for metric in ['avg_auc', 'bwt', 'fwt']:
            a = methods_dict['DP_FedEWC_CL'][metric]
            b = methods_dict['DP_FedEwc'][metric]
            w_stat, w_p = _wilcoxon(a, b)
            t_stat, t_p = _paired_ttest(a, b)
            sig_tests[ds_name][metric] = {
                'wilcoxon_stat': w_stat, 'wilcoxon_p': w_p,
                'ttest_stat': t_stat, 'ttest_p': t_p,
                'mean_diff': round(float(np.mean(a) - np.mean(b)), 4),
            }

    return summary, sig_tests, per_ds_method_metric


# ---------------------------------------------------------------------------
# Ablation, repeated over seeds (MIMIC-IV-Sim only)
# ---------------------------------------------------------------------------

def run_ablation_multiseed(n_seeds, verbose=True):
    base_seed = DATA_SEED_BASE['mimic_iv_sim']
    variant_metrics = {v: {'avg_auc': [], 'bwt': [], 'fwt': []}
                       for v in COMPONENT_VARIANTS}

    for s in range(n_seeds):
        data_seed = base_seed * 1000 + s
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        ds = generate_mimic_iv_sim(n_clients=N_CLIENTS, seed=data_seed)
        for vname, vfn in COMPONENT_VARIANTS.items():
            torch.manual_seed(2000 + s)
            R, eps = vfn(ds, BASE_CFG)
            m = compute_all_metrics(R, eps)
            variant_metrics[vname]['avg_auc'].append(m['avg_auc'])
            variant_metrics[vname]['bwt'].append(m['bwt'])
            variant_metrics[vname]['fwt'].append(m['fwt'])
        if verbose:
            print(f"  [ablation] seed {s+1}/{n_seeds} done", flush=True)

    summary = {v: {k: _summ(vals) for k, vals in metrics.items()}
              for v, metrics in variant_metrics.items()}

    # Significance: full vs no_recycling, and full vs no_fim_aggregation, on BWT
    sig = {}
    for comp in ['no_recycling', 'no_fim_aggregation', 'no_ewc']:
        a = variant_metrics['full_dp_fedewc_cl']['bwt']
        b = variant_metrics[comp]['bwt']
        w_stat, w_p = _wilcoxon(a, b)
        t_stat, t_p = _paired_ttest(a, b)
        sig[f'full_vs_{comp}_bwt'] = {
            'wilcoxon_stat': w_stat, 'wilcoxon_p': w_p,
            'ttest_stat': t_stat, 'ttest_p': t_p,
            'mean_diff': round(float(np.mean(a) - np.mean(b)), 4),
        }
    return summary, sig, variant_metrics


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--ablation-seeds', type=int, default=10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Running main comparison over {args.seeds} seeds …")
    summary, sig_tests, raw = run_main_comparison(args.seeds)
    print(f"Main comparison done in {time.perf_counter()-t0:.1f}s")

    with open('multiseed_results.json', 'w') as fh:
        json.dump({'summary': summary, 'significance': sig_tests,
                   'n_seeds': args.seeds}, fh, indent=2)
    print("Saved -> multiseed_results.json")

    t1 = time.perf_counter()
    print(f"\nRunning ablation over {args.ablation_seeds} seeds …")
    ab_summary, ab_sig, ab_raw = run_ablation_multiseed(args.ablation_seeds)
    print(f"Ablation done in {time.perf_counter()-t1:.1f}s")

    with open('multiseed_ablation.json', 'w') as fh:
        json.dump({'summary': ab_summary, 'significance': ab_sig,
                   'n_seeds': args.ablation_seeds}, fh, indent=2)
    print("Saved -> multiseed_ablation.json")

    print(f"\nTotal wall-clock: {time.perf_counter()-t0:.1f}s")

    # Print a compact summary
    print("\n" + "=" * 78)
    print("MAIN COMPARISON (mean ± std over seeds)")
    for ds_name in summary:
        print(f"\n-- {ds_name} --")
        for method in summary[ds_name]:
            m = summary[ds_name][method]
            print(f"  {method:<18} AUC={m['avg_auc']['mean']:.3f}±{m['avg_auc']['std']:.3f}  "
                  f"BWT={m['bwt']['mean']:+.4f}±{m['bwt']['std']:.4f}  "
                  f"FWT={m['fwt']['mean']:+.4f}±{m['fwt']['std']:.4f}")
        print(f"  Significance (DP-FedEWC-CL vs DP-FedEwc): {sig_tests[ds_name]}")
    print("=" * 78)


if __name__ == '__main__':
    main()
