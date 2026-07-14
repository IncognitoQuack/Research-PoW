# DP-FedEWC-CL — Reproduction Guide

This repository accompanies the revised manuscript submitted to JDSA. It
was updated in response to reviewer feedback requesting (1) more
realistic synthetic data, (2) multiple independent runs with standard
deviations and significance tests, and (3) a more convincing analysis of
the budget-recycling component.

## Requirements
```
pip install torch numpy scipy scikit-learn matplotlib --break-system-packages
```

## 1. Single-seed reference run (fast smoke test / point estimates)
Generates `all_results.json` + `ablation.json` (used for the illustrative,
single-seed sensitivity figures: privacy-utility sweep, lambda sweep, FIM
quality analytical figure).
```
python3 run_fast.py
```

## 2. Statistically rigorous evaluation (10-seed protocol — THE MAIN RESULT)
Generates `multiseed_results.json` (main comparison, mean/std/CI + paired
Wilcoxon and t-tests vs. DP-FedEwc) and `multiseed_ablation.json`
(component ablation, same protocol). This is what Table 1, Table 2,
Table 3, and Table 4 of the paper are built from.
```
python3 run_multiseed.py --seeds 10 --ablation-seeds 10
```
Wall-clock: approximately 25–30 minutes on a standard CPU (10 seeds x 3
datasets x 6 methods for the main comparison, plus 10 seeds x 4 variants
for the ablation).

## 3. Long-horizon budget-recycling diagnostic
Generates `longhorizon_results.json`: an 8-task extension of MIMIC-IV-Sim
with convergence-dependent early stopping, used to test the budget
recycling mechanism under conditions designed to satisfy its precondition
(see Remark 1 and Section 7.5 of the paper).
```
python3 run_longhorizon.py --seeds 10 --n_tasks 8
```

## 4. Regenerate figures
```
python3 generate_figures.py       # fig1_architecture, fig2_privacy_utility,
                                   # fig5_fim_quality (single-seed, illustrative)
python3 generate_figures_v2.py    # fig3_ablation, fig4_forgetting (10-seed,
                                   # with error bars + significance stars),
                                   # fig6_longhorizon (new)
```
Note: `generate_figures.py` also emits `fig2_main_results.pdf`,
`fig4_ablation.pdf`, `fig5_forgetting.pdf`, and `fig6_fim_quality.pdf`
under its own internal numbering; these were renamed to
`fig2_privacy_utility.pdf`, `fig3_ablation.pdf` (superseded by the v2
script), `fig4_forgetting.pdf` (superseded), and `fig5_fim_quality.pdf`
respectively to match the filenames referenced in `main.tex`.

## File overview
- `data_generator.py` — the three benchmarks (MIMIC-IV-Sim, eICU-Sim,
  HiRID-Sim), now including a realism layer: missingness, coding noise,
  label uncertainty, and per-client correlation/volume heterogeneity.
- `data_generator_longhorizon.py` — 8-task extension used only for the
  budget-recycling diagnostic.
- `model.py` — MLP, DP-SGD, Fisher computation/aggregation, EWC penalty,
  privacy accounting.
- `baselines.py` — FedAvg, DP-FedAvg, Local-EWC, DP-FedEwc,
  Centralised-EWC.
- `run_fast.py` — proposed method (DP-FedEWC-CL) + single-seed driver.
- `run_multiseed.py` — 10-seed statistical evaluation protocol with
  paired significance testing.
- `run_longhorizon.py` — long-horizon budget-recycling diagnostic with
  convergence-dependent early stopping.
- `metrics.py` — avg-AUC, BWT, FWT computation.
- `generate_figures.py` / `generate_figures_v2.py` — figure generation.

All scripts are seeded and deterministic given the same seed; the 10-seed
protocol uses seeds documented directly in `run_multiseed.py` /
`run_longhorizon.py`.
