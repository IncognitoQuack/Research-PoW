"""Utility functions: seeding, caching, logging."""

import os
import random
import logging
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def cache_path(cache_dir: str, key: str) -> Path:
    """Return a deterministic cache file path for a string key."""
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{h}.pkl"


def save_cache(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_cache(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dirs(*dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
