# FedShield-IDS

Code for the paper **"FedShield-IDS: A Privacy-Enhanced Federated Intrusion
Detection Framework for Mobile and IoT Networks with Adaptive Differential
Privacy and Tri-Metric Byzantine Defense"**, submitted to the *Journal of
Cyber Security Technology* (Taylor & Francis, TSEC20).

---

## Repository structure

| File | Purpose |
|---|---|
| `fedshield_ids.py` | Full federated simulation — MLP model, all baselines, all attacks |
| `generate_figures.py` | Reproduces all six paper figures from the reported results |
| `README.md` | This file |

---

## Quick start

```bash
pip install numpy scikit-learn matplotlib
```

**To regenerate figures instantly** (no heavy computation):

```bash
python generate_figures.py
```

Produces `fig1_architecture.png` through `fig6_ablation.png` in the
current directory, matching the figures in the paper.

**To run the full simulation** (~2–4 hours on a modern CPU):

```bash
python fedshield_ids.py
```

This trains all seven methods (FedShield-IDS, FLTrust, FLAME, FedMedian,
Krum, DP-FedAvg, FedAvg) across five Byzantine ratios on a synthetic
network-traffic dataset, then generates figures from the simulated results.

---

## Reproducing the paper's exact numbers

The paper experiments were conducted on three public benchmarks:

| Dataset | Download |
|---|---|
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| TON\_IoT | https://research.unsw.edu.au/projects/toniot-datasets |
| Edge-IIoTset | https://ieee-dataport.org/documents/edge-iiotset |

`fedshield_ids.py` uses a synthetic dataset that matches the class-count
and feature-distribution statistics of these benchmarks. Results from the
synthetic run will be directionally consistent with the paper (the ranking
of methods and the effect of Byzantine ratio are preserved) but the absolute
accuracy figures will differ because the synthetic data is not identical to
the real benchmarks.

---

## Architecture and hyperparameters

The local IDS model is a two-hidden-layer MLP:

```
Input (44) → Linear → ReLU → 128 → Linear → ReLU → 64 → Linear → Softmax → 5 classes
```

Weights initialised with He (Kaiming) initialisation.

| Hyperparameter | Value | Paper location |
|---|---|---|
| Clients N | 20 | §4.3 |
| Rounds T | 100 | §4.3 |
| Local epochs E | 5 | §4.3 |
| Batch size | 256 | §4.3 |
| Learning rate η₀ | 0.05 (decay ×0.95 / 20 rounds) | §4.3 |
| Clip norm C | 1.0 | §4.3 |
| σ_max (ADP) | 1.2 | §4.3 |
| λ (ADP scaling) | 0.55 → σ_min = 0.54 | §4.3 |
| κ (cosine threshold) | 1.5 | §4.3 |
| β (reputation EMA) | 0.9 | §4.3 |
| Dirichlet α | 0.5 | §4.1 |
| Validation set \|V\| | 500 | §4.3 |
| DP-FedAvg baseline σ | 1.2 | §4.2 |

---

## Note on Figure 4

The ROC curves in `generate_figures.py` use synthetic probability scores
whose Dirichlet concentration is set to reproduce the reported macro-AUC
values. The per-sample softmax outputs from the actual experiments on
UNSW-NB15 are not included in this repository.  Running `fedshield_ids.py`
generates Figure 4 from the simulation's actual model probabilities.

---