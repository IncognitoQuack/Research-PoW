# CT-GNN — Reproduction Guide

## Requirements
```
pip install torch numpy scipy scikit-learn matplotlib
```

## Run experiments (generates all_results.json + ablation.json)
```
python3 run_fast.py
```

## Regenerate figures (requires all_results.json)
```
python3 generate_figures.py
```

## Files
| File | Purpose |
|---|---|
| `data_generator.py` | Synthetic IoT benchmark generator (SWaT/WADI/PSM-style) |
| `granger_graph.py` | Granger-causality graph construction |
| `ct_gnn_model.py` | CT-GNN model (TCE + CTA + CGAT + Propagation Scorer) |
| `baselines.py` | MTAD-GAT, GANF, LSTM-VAE baselines |
| `metrics.py` | PA-F1, RCL@k, latency measurement |
| `run_fast.py` | Full experiment runner |
| `generate_figures.py` | All 6 paper figures |
