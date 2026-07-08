#!/usr/bin/env bash
# Orchestrates the full pipeline for one dataset.
# Automatically uses configs/config_<dataset>.yaml if it exists (e.g. a
# compute-budget-tuned config for large datasets like CIC-IDS2017),
# otherwise falls back to configs/config.yaml.
#
# Usage: bash run_all.sh <nsl_kdd|cic_ids2017> [n_seeds]
set -euo pipefail

DATASET="${1:?Usage: run_all.sh <nsl_kdd|cic_ids2017> [n_seeds]}"
N_SEEDS="${2:-1}"
RAW_DIR="data/raw/${DATASET}"

DATASET_CONFIG="configs/config_${DATASET}.yaml"
if [ -f "${DATASET_CONFIG}" ]; then
  CONFIG="${DATASET_CONFIG}"
  echo ">>> Using dataset-specific config: ${CONFIG}"
else
  CONFIG="configs/config.yaml"
  echo ">>> Using default config: ${CONFIG}"
fi

MAX_TRAIN_SAMPLES=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
v = cfg.get('data', {}).get('max_train_samples')
print(v if v is not None else '')
")

echo "=== [1/6] Preprocessing ${DATASET} ==="
if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
  echo ">>> Subsampling training set to ${MAX_TRAIN_SAMPLES} rows (stratified; test set stays full)"
  python3 src/data_utils.py --dataset "${DATASET}" --raw_dir "${RAW_DIR}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}"
else
  python3 src/data_utils.py --dataset "${DATASET}" --raw_dir "${RAW_DIR}"
fi

for SEED in $(seq 1 "${N_SEEDS}"); do
  echo "=== [2/6] Training deep models (seed=${SEED}) ==="
  python3 src/train.py --dataset "${DATASET}" --config "${CONFIG}" --model ensemble --seed "${SEED}"
  python3 src/train.py --dataset "${DATASET}" --config "${CONFIG}" --model cnn_only --seed "${SEED}"
  python3 src/train.py --dataset "${DATASET}" --config "${CONFIG}" --model rnn_only --seed "${SEED}"
done

echo "=== [3/6] Training classical baselines ==="
python3 src/baselines.py --dataset "${DATASET}"

echo "=== [4/6] Evaluating all models ==="
python3 src/evaluate.py --dataset "${DATASET}" --config "${CONFIG}"

echo "=== [5/6] Benchmarking inference latency ==="
python3 src/latency_bench.py --dataset "${DATASET}" --config "${CONFIG}"

echo "=== [6/6] Generating figures ==="
python3 src/make_figures.py --dataset "${DATASET}"

echo "Done. See results/ and figures/."
