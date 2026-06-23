# CipherTune 

This is the real, runnable implementation behind the CipherTune paper
(Privacy-Preserving LLM Fine-Tuning Framework Using Homomorphic Encryption
for Secure Cybersecurity Applications). Every number that ends up in the
paper's tables and figures comes from running this code on a local machine
machine and all the resulting `results/` JSON files back -- nothing
in the manuscript is fabricated or synthetic.

## What this code does

Three conditions, same model, same data splits, same local-training
budget, compared head to head:

1. **Centralized** -- all 8 simulated "organizations'" CVE data pooled,
   standard LoRA fine-tuning, no privacy mechanism (upper-bound reference).
2. **DP-FedAvg** -- federated LoRA averaging with client-level differential
   privacy (clip + Gaussian noise), swept over 3 epsilon operating points.
3. **CipherTune (ours)** -- federated LoRA averaging where every client's
   update is quantized and Paillier-encrypted before it ever leaves the
   client; the server only ever decrypts the cohort-level sum.

All three are then probed with a loss-threshold membership-inference
attack (Yeom et al., 2018) to get a concrete privacy number (AUC).

## Setup

```bash
cd ciphertune
pip install -r requirements.txt
```

If you have a GPU, PyTorch will use it automatically; everything also
runs on CPU with the default (laptop-friendly) sizes in `config.py`.

## Run order (do these in sequence)

### 1. Fetch real CVE data from NVD

```bash
python -m ciphertune.data_fetch
```

This hits the real NVD REST API v2.0 and pulls real, public-domain CVE
records (2023 through today), keeping only records that have an official
NVD CVSS v3.x severity score. Takes a few minutes (NVD rate-limits
unauthenticated requests to 5 per 30 seconds). Optional: set
`NVD_API_KEY` in your environment for a faster, authenticated pull (free
sign-up at https://nvd.nist.gov/developers/request-an-api-key).

Writes `data/cve_dataset.csv` and prints the class balance and
description-length statistics -- please paste that console output back to
me along with the results files, it's useful for the paper's dataset
description.

### 2. Partition into federated client shards

```bash
python -m ciphertune.partition
```

Builds the held-out test set and the 8 client shards. `run_all.py` will
call this automatically if you skip it, but running it standalone lets
you sanity-check the printed class balance first.

### 3. Run everything

```bash
python -m ciphertune.run_all
```

This runs all three conditions (centralized, the DP epsilon sweep, and
CipherTune), runs the membership-inference attack on each, and writes:

- `results/centralized.json`
- `results/dp_fedavg_eps0.5.json`, `results/dp_fedavg_eps2.0.json`, `results/dp_fedavg_eps8.0.json`
- `results/ciphertune.json`
- `results/mia_summary.json`
- `results/results.json` (everything consolidated)
- `results/summary_table.csv` (the table that maps directly to the paper)


### 4. generate figures

```bash
python -m ciphertune.make_figures
```

## Config you might want to change

Everything lives in `ciphertune/config.py`, with "SCALE UP" comments
marking the values most worth increasing if you have a stronger machine
(`TARGET_DATASET_SIZE`, `FEDERATED_ROUNDS`, `NUM_CLIENTS`,
`LORA_TARGET_LAYER_INDICES`, `PAILLIER_KEY_BITS`).
