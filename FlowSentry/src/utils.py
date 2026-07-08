import os
import random
import json

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(priority=("mps", "cuda", "cpu")):
    for p in priority:
        if p == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    ensure_dirs(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)
