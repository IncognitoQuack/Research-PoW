# CipherTune -- Phase 2 code package

This is the real, runnable implementation behind the CipherTune paper
(Privacy-Preserving LLM Fine-Tuning Framework Using Homomorphic Encryption
for Secure Cybersecurity Applications). Every number that ends up in the
paper's tables and figures comes from YOU running this code on YOUR
machine and sending the resulting `results/` JSON files back -- nothing
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

## Why you have to run the data fetch and training yourself

I (Claude) do not have network access to `services.nvd.nist.gov` or to
Hugging Face's model hub from my sandbox, and the full bert-base-uncased
training run is too compute-heavy to run inside a chat session anyway.
Every line of logic in this package has already been unit-tested against
hand-built inputs that match the real schemas/APIs it depends on (see the
self-tests below), but the *actual* real-world execution -- the real NVD
pull and the real model training -- has to happen on your machine.

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

**Please send me back the full console output of this run, plus the
contents of the `results/` folder.** That console output is exactly what
gets quoted/summarized in the paper's Experimental Setup and Results
sections, so the more complete it is, the better.

### 4. (After I confirm the results look right) generate figures

```bash
python -m ciphertune.make_figures
```

This refuses to run until `results/results.json` exists, and only ever
plots numbers from that file -- there is no path by which a placeholder
or synthetic number can end up in a figure. I'll likely ask you to run
this after we've reviewed the numbers together, but the script is ready
now if you want to see it immediately.

## What to expect (so the numbers don't surprise you)

- **DP-FedAvg will likely look bad, especially at the tighter epsilon
  values.** This isn't a bug. Client-level differential privacy (the
  McMahan et al. 2018 / Geyer et al. 2017 construction used here)
  fundamentally needs many participants per round to preserve utility at
  a meaningful epsilon; with only 8 simulated organizations (realistic
  for inter-organizational threat-intel sharing, as opposed to a
  millions-of-users mobile-keyboard setting), the noise needed to hit a
  tight epsilon will likely swamp the signal. This is itself a real,
  citable, and actually useful finding for the paper -- it's part of the
  argument for why an encryption-based approach is attractive in exactly
  this small-cohort regime.
- **CipherTune will be slower per round than DP-FedAvg or Centralized.**
  Paillier encryption/decryption is genuinely expensive; the overhead
  numbers (encryption time, decryption time, bytes transmitted) are
  exactly what Objective 2 asks us to measure and report honestly.
- All of this is fine and expected -- send me whatever numbers actually
  come out, and we'll write the paper around what's real.

## Config you might want to change

Everything lives in `ciphertune/config.py`, with "SCALE UP" comments
marking the values most worth increasing if you have a stronger machine
(`TARGET_DATASET_SIZE`, `FEDERATED_ROUNDS`, `NUM_CLIENTS`,
`LORA_TARGET_LAYER_INDICES`, `PAILLIER_KEY_BITS`).

## Self-tests already run on this code (by me, before sending it to you)

- `ciphertune/crypto_utils.py` -- `python -m ciphertune.crypto_utils` runs
  a self-contained round-trip test of the full pack -> encrypt ->
  homomorphic-sum -> decrypt -> unpack -> dequantize pipeline against
  synthetic vectors and asserts the recovered average matches the true
  average within quantization tolerance. Already verified passing.
- `ciphertune/dp_utils.py` -- `python -m ciphertune.dp_utils` prints the
  noise-vs-signal sweep used to choose the epsilon operating points.
- Full pipeline integration: every module has been wired together and
  run end-to-end (centralized + DP sweep + CipherTune + MIA + summary
  table) using a tiny local stand-in model and a 200-row synthetic
  placeholder dataset, purely to confirm there are no shape mismatches,
  import errors, or JSON-serialization bugs. That test data is NOT
  included in this package and was never used for anything paper-related.
