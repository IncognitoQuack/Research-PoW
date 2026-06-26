"""Frequency branch: 8×8 block DCT feature extraction for quantum kernel input."""

from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .utils import get_logger, cache_path, save_cache, load_cache

log = get_logger(__name__)

_BLOCK = 8


def _dct_features_single(path: Path, n_components: int = 8) -> np.ndarray:
    """
    Compute mean-magnitude AC DCT features from 8×8 blocks.

    Steps:
      1. Load greyscale, resize to 224×224.
      2. Tile into non-overlapping 8×8 blocks.
      3. Apply 2-D DCT to each block.
      4. Collect the 63 AC coefficients (skip DC).
      5. Average across blocks; return top-n by mean magnitude.

    Normalise to zero mean / unit variance across the dataset (done in caller).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(n_components, dtype=np.float32)

    img = cv2.resize(img, (224, 224)).astype(np.float32)
    h, w = img.shape
    ac_accum = np.zeros(63, dtype=np.float64)
    n_blocks = 0

    for r in range(0, h - _BLOCK + 1, _BLOCK):
        for c in range(0, w - _BLOCK + 1, _BLOCK):
            patch = img[r : r + _BLOCK, c : c + _BLOCK]
            dct = cv2.dct(patch)
            ac = np.abs(dct.flatten()[1:])   # skip DC
            ac_accum += ac
            n_blocks += 1

    if n_blocks == 0:
        return np.zeros(n_components, dtype=np.float32)

    mean_ac = ac_accum / n_blocks
    top_idx = np.argsort(mean_ac)[::-1][:n_components]
    return mean_ac[top_idx].astype(np.float32)


def extract_dct_features(
    image_paths: List[Path],
    n_components: int = 8,
    cache_dir: str = "outputs/cache",
    split_tag: str = "train",
) -> np.ndarray:
    """
    Compute DCT frequency features for all images.

    Returns
    -------
    features : ndarray of shape (n, n_components)
    """
    key = f"dct_{split_tag}_{n_components}_{len(image_paths)}"
    cp = cache_path(cache_dir, key)
    if cp.exists():
        log.info("Loading cached DCT features for split=%s", split_tag)
        return load_cache(cp)

    log.info("Extracting DCT features (%d-component) for %d images (split=%s) …",
             n_components, len(image_paths), split_tag)
    out = np.array([
        _dct_features_single(p, n_components)
        for p in tqdm(image_paths, desc=f"DCT [{split_tag}]", unit="img")
    ], dtype=np.float32)

    save_cache(out, cp)
    log.info("Saved DCT features → %s", cp)
    return out
