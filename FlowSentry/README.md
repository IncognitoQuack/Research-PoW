# Hybrid CNN–RNN Ensemble for Real-Time Network Anomaly Detection

Reproducible research code for a hybrid CNN + RNN (Bi-LSTM) ensemble network
intrusion / zero-day anomaly detector, benchmarked on **NSL-KDD** and
**CIC-IDS2017**, with a latency comparison against a rule/signature-based
baseline and classical ML baselines (Random Forest, SVM).

This repository is built to produce **real, reproducible numbers** — nothing
in `results/` is pre-populated. You run the pipeline on the actual datasets
and the metrics/figures are generated from that run.

---

## 1. Repository layout

```
nids-ensemble/
├── configs/
│   └── config.yaml           # all hyperparameters in one place
├── data/
│   ├── raw/                  # put downloaded raw CSVs here (see §2)
│   └── processed/            # cache of cleaned/encoded arrays (auto-created)
├── src/
│   ├── data_utils.py         # loading, cleaning, encoding, scaling, labeling
│   ├── datasets.py           # PyTorch Dataset/window builders
│   ├── models.py             # CNN branch, RNN branch, fusion ensemble
│   ├── baselines.py          # RF / SVM / signature-rule baseline
│   ├── train.py              # training entry point (CLI)
│   ├── evaluate.py           # metrics: Acc, Prec, Rec, F1, FPR, AUC, CM
│   ├── latency_bench.py      # inference-latency benchmarking (edge sim.)
│   └── make_figures.py       # generates every figure used in the paper
├── tests/
│   └── smoke_test.py         # synthetic-data end-to-end pipeline check
├── run_all.sh                # orchestrates the full experiment
├── requirements.txt
└── README.md
```

## 2. Getting the real datasets (you must download these yourself)

The code will not fabricate or substitute data. Download the official files
and place them as below.

### NSL-KDD
Source: Canadian Institute for Cybersecurity, University of New Brunswick
- https://www.unb.ca/cic/datasets/nsl.html
Download `KDDTrain+.txt` and `KDDTest+.txt`, place them at:
```
data/raw/nsl_kdd/KDDTrain+.txt
data/raw/nsl_kdd/KDDTest+.txt
```

### CIC-IDS2017
Source: Canadian Institute for Cybersecurity, University of New Brunswick
- https://www.unb.ca/cic/datasets/ids-2017.html
Download the "MachineLearningCSV" (flow-labeled CSVs, one per day/attack),
place all CSVs at:
```
data/raw/cic_ids2017/*.csv
```

> These datasets require a data-usage request/agreement from UNB CIC in some
> distributions — follow their site's instructions. This is standard for
> both datasets and is why they are not bundled here.

## 3. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12, PyTorch 2.x (CPU or CUDA — CUDA auto-detected).

## 4. Quick correctness check (no real data needed)

Before touching the real datasets, verify the whole pipeline is wired
correctly using a synthetic stand-in dataset with the same schema:

```bash
python3 tests/smoke_test.py
```

This should complete in under a minute and print PASS for every stage
(loading → windowing → model forward pass → training loop → evaluation →
latency benchmark → figure generation). It proves there are no shape/logic
bugs; it does **not** produce meaningful detection numbers — it's a plumbing
test, not a substitute for the real experiment. Discard `results/smoke_*`.

## 5. Running the real experiment

```bash
# 1) Preprocess (run once per dataset; caches to data/processed/)
python3 src/data_utils.py --dataset nsl_kdd --raw_dir data/raw/nsl_kdd
python3 src/data_utils.py --dataset cic_ids2017 --raw_dir data/raw/cic_ids2017

# 2) Train the CNN-RNN ensemble + single-model baselines
python3 src/train.py --dataset nsl_kdd --config configs/config.yaml
python3 src/train.py --dataset cic_ids2017 --config configs/config_cic_ids2017.yaml

# 3) Train classical-ML baselines (RF, SVM) for comparison
python3 src/baselines.py --dataset nsl_kdd
python3 src/baselines.py --dataset cic_ids2017

# 4) Evaluate everything (writes results/*.json + results/*.csv)
python3 src/evaluate.py --dataset nsl_kdd
python3 src/evaluate.py --dataset cic_ids2017

# 5) Benchmark inference latency (ensemble vs. baselines, batch=1 "edge" mode)
python3 src/latency_bench.py --dataset nsl_kdd

# 6) Generate all figures used in the paper (reads results/*.json)
python3 src/make_figures.py --dataset nsl_kdd

# or just, per dataset (auto-picks configs/config_<dataset>.yaml if it exists):
bash run_all.sh nsl_kdd
bash run_all.sh cic_ids2017
```

Outputs:
- `results/<dataset>_metrics.json` — accuracy, precision, recall, F1, FPR, ROC-AUC per model
- `results/<dataset>_confusion_matrix.csv`
- `results/<dataset>_latency.json` — mean/median/p95 inference latency per model
- `figures/*.png` — training curves, ROC curves, confusion matrix heatmap,
  latency comparison bar chart, ablation bar chart

## 5b. Compute budget: CIC-IDS2017 is ~18x larger than NSL-KDD

NSL-KDD has ~126k training rows; CIC-IDS2017 has ~2.26M. With identical
hyperparameters, CIC-IDS2017 would take roughly an order of magnitude longer
per epoch, and full multi-day runs on a laptop are common if nothing is
tuned for it. `run_all.sh` automatically uses
**`configs/config_cic_ids2017.yaml`** for that dataset, which differs from
the default config only in ways that affect wall-clock time, not the model
architecture itself:

- **Apple Silicon GPU (MPS) support**: `get_device()` now tries `mps` before
  falling back to `cuda`/`cpu`. On an M-series Mac this alone can give a
  large speedup for the CNN/LSTM branches. No action needed — it's
  automatic; if you ever hit an MPS op that isn't implemented, PyTorch will
  print a clear error naming the op (rare in recent versions).
- **Larger batch size** (1024 vs. 256) — far fewer optimizer steps per epoch.
- **Fewer max epochs** (15 vs. 40) with tighter early-stopping patience (3
  vs. 6) — justified because 2.26M rows already give many gradient updates
  per epoch, so full convergence needs fewer passes over the data.
- **`num_workers: 4`** for parallel DataLoader batching (set to `0` in
  `configs/config.yaml` if you ever see multiprocessing issues — 0 is
  always safe, just single-threaded).
- **Optional stratified training-set subsampling** via `data.max_train_samples`
  (default 300,000 in the CIC config). This only caps the **training**
  split; the **test** split is never subsampled, so reported metrics are
  still evaluated against the full, honest test distribution. This is a
  legitimate, common practice for making a huge dataset tractable on
  commodity hardware — but you must disclose it if you use it, e.g.:
  *"CIC-IDS2017 models were trained on a stratified 300,000-row subsample
  of the ~2.26M available training flows for computational tractability;
  evaluation was performed on the full held-out test split."*
  To use the full 2.26M rows instead, set `max_train_samples: null` in
  `configs/config_cic_ids2017.yaml` (expect it to take much longer).

You'll also now see a live progress bar per epoch (via `tqdm`) plus a
printed steps-per-epoch count and an ETA estimate after epoch 1, so a long
run is visibly progressing rather than looking frozen.

## 6. Reporting real results in the paper

Only use the numbers written by `evaluate.py` / `latency_bench.py` into
`results/`. Do not hand-edit these files. If you re-run with different
seeds/hyperparameters, re-run `make_figures.py` so the figures match.

## 7. Random seeds & determinism

All scripts accept `--seed` (default 42) and set it for `numpy`, `random`,
and `torch` (including CUDA, with `torch.use_deterministic_algorithms(True)`
where supported). Minor nondeterminism can still occur on GPU due to cuDNN
kernels; report mean ± std over at least 5 seeds for the paper's headline
numbers (the `run_all.sh` script has a `--n_seeds` flag for this).

## 8. Citation

If you use this code, please cite the accompanying paper (details to be
added once accepted/published).
