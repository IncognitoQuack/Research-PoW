# NeuroSym-ZTA: Neuro-Symbolic Zero-Trust Authentication

**Paper**: "NeuroSym-ZTA: A Neuro-Symbolic Framework for Explainable Authentication and Access Control in Zero-Trust Network Architectures"
**Journal**: Journal of Information Security and Applications (JISA), Elsevier

---

## What this is

NeuroSym-ZTA combines a **BiGRU behavioural encoder** with a **differentiable symbolic ZTA policy layer** whose rule weights are learned via backpropagation. An entropy-based adaptive fusion gate shifts the decision weight toward symbolic rules when the neural component is uncertain. Every decision produces a **dual-channel explanation**: gradient-times-input attribution (neural path) and a rule-activation trace (symbolic path).

---

## Dataset

NSL-KDD 

| File | Records |
|------|---------|
| KDDTrain+.txt | 125,973 |
| KDDTest+.txt  | 22,544  |

**ZTA class mapping:** `normal` → ALLOW (0) · DoS/Probe → CHALLENGE (1) · R2L/U2R → DENY (2)

---

## Setup

```bash
pip install torch scikit-learn shap matplotlib seaborn pandas numpy
```

Python 3.9+. GPU optional; CPU runtime ~10-15 min.

---

## Run

```bash
python neurosym_zta.py          # full run (30 epochs)
python neurosym_zta.py --quick  # smoke test (8 epochs, ~3 min)
```

---

## Results

| Method | Acc (%) | F1 (%) | DENY-F1 (%) |
|--------|---------|--------|-------------|
| **NeuroSym-ZTA (Ours)** | **77.86** | **65.08** | 31.73 |
| Static Fusion (a=0.5) | 73.49 | 59.63 | 23.82 |
| Standalone BiGRU      | 76.28 | 60.64 | 20.81 |
| Random Forest         | 49.61 | 32.51 |  0.00 |
| Rule Engine Only      | 43.29 | 20.14 |  0.00 |

DENY = R2L + U2R attacks (privilege escalation). NeuroSym-ZTA achieves **52.5% relative improvement** over BiGRU on DENY-F1. CPU latency: **34.6 ms** for 1,000 sessions.

---

## Symbolic Rule Definitions

| Rule | Feature | ZTA Signal |
|------|---------|-----------|
| R1 | `num_failed_logins / 5` | Brute-force authentication |
| R2 | `(root_shell + su_attempted) / 2` | Privilege escalation (U2R) |
| R3 | `serror_rate` | SYN error / scanning |
| R4 | `diff_srv_rate` | Lateral movement |
| R5 | `num_compromised / 10` | Host compromise |

Weights initialised at 1.0; **learned end-to-end** by backpropagation.

---  
