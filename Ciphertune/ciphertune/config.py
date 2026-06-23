"""
config.py
---------
Single source of truth for every hyperparameter used across the three
experimental conditions (Centralized, DP-FedAvg, CipherTune). Keeping all
of these in one place guarantees that the three conditions are evaluated
on an identical model, identical data splits and identical local-training
budget, which is the only way the comparison in Objective 2 is fair.

Defaults are deliberately chosen to be runnable on a single laptop
(CPU or one consumer GPU) in well under an hour per condition. If you
have access to a stronger GPU, the values flagged "SCALE UP" below are
the ones worth increasing first.
"""

import random
import numpy as np
import torch

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
NVD_PUBLISHED_START = "2026-03-01T00:00:00.000"   # single ~110-day window, safely under NVD's
NVD_PUBLISHED_END = "2026-06-19T00:00:00.000"      # 120-day hard limit -- no multi-window chunking
                                                    # is needed at this size, which removes an entire
                                                    # class of potential bugs. Still real, current
                                                    # ("2026 security tasks") CVE data, just a shorter,
                                                    # lighter-weight calendar slice. SCALE UP: widen
                                                    # (keeping any single span <=119 days, or rely on
                                                    # the automatic windowing in data_fetch.py for
                                                    # longer ranges) for a larger corpus.
NVD_MAX_WINDOW_DAYS = 119                          # NVD API hard limit is 120 consecutive days
                                                    # per pubStartDate/pubEndDate query; 119 is a
                                                    # 1-day safety margin against off-by-one issues.
RESULTS_PER_PAGE = 2000                            # NVD API's documented max per page
TARGET_DATASET_SIZE = 1200                         # decent, real, fast-to-train corpus size.
                                                    # SCALE UP: raise for a larger corpus.
MAX_TOKEN_LENGTH = 128                              # CVE descriptions are short; verified empirically in data_fetch.py

RAW_JSON_DIR = "data/raw_nvd"
PROCESSED_CSV = "data/cve_dataset.csv"
SPLITS_DIR = "data/splits"
TEST_FRACTION = 0.15
NUM_CLIENTS = 4                                     # SCALE UP: more clients = more realistic federation, slower wall-clock

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID = {l: i for i, l in enumerate(SEVERITY_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ----------------------------------------------------------------------
# Model / LoRA
# ----------------------------------------------------------------------
BASE_MODEL_NAME = "bert-base-uncased"
LORA_R = 4                                          # SCALE UP: 8 or 16 with more compute
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]            # attention Q/V projections only
LORA_TARGET_LAYER_INDICES = [11]                    # a single layer is enough to demonstrate the
                                                     # protocol end to end and keeps both local-training
                                                     # time and the number of Paillier-encrypted slots
                                                     # small. SCALE UP: add more layer indices, e.g.
                                                     # [9, 10, 11] or list(range(12)), with more compute.
NUM_LABELS = len(SEVERITY_LABELS)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
BATCH_SIZE = 16
LOCAL_EPOCHS_PER_ROUND = 1
FEDERATED_ROUNDS = 3                                # SCALE UP: 5-10 rounds with more compute
LEARNING_RATE = 2e-4
CENTRALIZED_EPOCHS = 3                              # centralized baseline trains the same total #epochs as FL (rounds*local_epochs)

# ----------------------------------------------------------------------
# Quantization (shared by DP clipping reference scale and CipherTune)
# ----------------------------------------------------------------------
QUANT_BITS = 16                                     # per-value fixed-point width before packing
QUANT_SCALE = 2 ** 12                               # multiply float delta by this before rounding to int

# ----------------------------------------------------------------------
# Differential privacy (DP-FedAvg baseline; client-level DP, McMahan et al. 2018 /
# Geyer et al. 2017 construction -- central single noise-addition to the
# client-averaged update; see dp_utils.py for the full design rationale)
# ----------------------------------------------------------------------
DP_CLIP_NORM = 1.0                      # S : L2-norm clip applied to each client's local update vector
DP_DELTA_PER_ROUND = 1e-5
DP_EPSILON_ROUND_SWEEP = [2.0]            # a single, moderate operating point. SCALE UP: add more
                                           # points e.g. [0.5, 2.0, 8.0] for a fuller Pareto-curve
                                           # ablation if you want a richer sensitivity analysis later.
DP_HEADLINE_EPSILON_ROUND = 2.0           # which sweep point appears in the main 3-way results table

# ----------------------------------------------------------------------
# Paillier homomorphic encryption (CipherTune)
# ----------------------------------------------------------------------
PAILLIER_KEY_BITS = 1024    # research-prototype size for tractable wall-clock benchmarking
                            # (see Limitations: production deployment should use >=2048-bit keys)
PACK_MARGIN_BITS = 4        # extra headroom bits per slot to absorb the sum over NUM_CLIENTS without overflow

# ----------------------------------------------------------------------
# Membership inference attack
# ----------------------------------------------------------------------
MIA_SAMPLE_SIZE = 150       # number of member / non-member examples used to estimate attack AUC
                             # (kept comfortably below the held-out test set size at the reduced
                             # TARGET_DATASET_SIZE, so member/non-member sample counts stay balanced)
