"""Classical spatial branch: EfficientNet-B4 feature extraction."""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from .utils import get_logger, cache_path, save_cache, load_cache

log = get_logger(__name__)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def _build_backbone(device: torch.device) -> nn.Module:
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)
    model.classifier = nn.Identity()   # strip head → 1792-dim avgpool output
    model.eval()
    return model.to(device)


def extract_classical_features(
    image_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
    cache_dir: str = "outputs/cache",
    split_tag: str = "train",
) -> np.ndarray:
    """
    Extract 1792-dim EfficientNet-B4 feature vectors.

    Results are cached to disk; subsequent calls return immediately.

    Returns
    -------
    features : ndarray of shape (n, 1792)
    """
    key = f"classical_{split_tag}_{len(image_paths)}"
    cp = cache_path(cache_dir, key)
    if cp.exists():
        log.info("Loading cached classical features for split=%s", split_tag)
        return load_cache(cp)

    log.info("Extracting EfficientNet-B4 features for %d images (split=%s) …",
             len(image_paths), split_tag)
    model = _build_backbone(device)
    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size),
                      desc=f"EfficientNet [{split_tag}]", unit="batch"):
            batch = image_paths[i : i + batch_size]
            tensors = []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(_transform(img))
                except Exception:
                    tensors.append(torch.zeros(3, 224, 224))
            batch_t = torch.stack(tensors).to(device)
            feat = model(batch_t).cpu().numpy()
            features.append(feat)

    out = np.vstack(features)
    save_cache(out, cp)
    log.info("Saved classical features → %s", cp)
    return out
