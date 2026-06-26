"""Dataset utilities: load image paths and labels from real/fake folder layout."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_split(
    split_dir: str,
    max_per_class: int = None,
    seed: int = 42,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """
    Load image paths and integer labels from a split directory.

    Expected layout:
        split_dir/
          real/   (label 0)
          fake/   (label 1)

    For attribution, fake sub-folders may be named after manipulation types.
    Only real/ and fake/ are required for binary detection.

    Returns
    -------
    paths  : list of Path
    labels : int ndarray  (0 = real, 1 = fake)
    names  : list of str  class names per sample (for attribution)
    """
    split_dir = Path(split_dir)
    rng = np.random.default_rng(seed)

    paths, labels, names = [], [], []

    real_dir = split_dir / "real"
    if real_dir.is_dir():
        real_imgs = sorted(
            p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
        )
        if max_per_class and len(real_imgs) > max_per_class:
            idx = rng.choice(len(real_imgs), max_per_class, replace=False)
            real_imgs = [real_imgs[i] for i in sorted(idx)]
        paths.extend(real_imgs)
        labels.extend([0] * len(real_imgs))
        names.extend(["Real"] * len(real_imgs))

    fake_dir = split_dir / "fake"
    if fake_dir.is_dir():
        # Collect directly from fake/ or from sub-folders (manipulation types)
        sub_dirs = [d for d in fake_dir.iterdir() if d.is_dir()]
        if sub_dirs:
            # e.g. fake/DeepFakes/, fake/Face2Face/ …
            for sub in sorted(sub_dirs):
                imgs = sorted(
                    p for p in sub.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
                )
                cap = max_per_class // max(len(sub_dirs), 1) if max_per_class else None
                if cap and len(imgs) > cap:
                    idx = rng.choice(len(imgs), cap, replace=False)
                    imgs = [imgs[i] for i in sorted(idx)]
                paths.extend(imgs)
                labels.extend([1] * len(imgs))
                names.extend([sub.name] * len(imgs))
        else:
            fake_imgs = sorted(
                p for p in fake_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
            )
            if max_per_class and len(fake_imgs) > max_per_class:
                idx = rng.choice(len(fake_imgs), max_per_class, replace=False)
                fake_imgs = [fake_imgs[i] for i in sorted(idx)]
            paths.extend(fake_imgs)
            labels.extend([1] * len(fake_imgs))
            names.extend(["Fake"] * len(fake_imgs))

    labels = np.array(labels, dtype=np.int64)
    return paths, labels, names


def pil_load(path: Path, size: int = 224) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return img


def split_summary(paths, labels, names=None) -> str:
    real = int((labels == 0).sum())
    fake = int((labels == 1).sum())
    msg = f"  total={len(paths)}  real={real}  fake={fake}"
    if names:
        from collections import Counter
        counts = Counter(n for n, l in zip(names, labels) if l == 1)
        msg += "  fake_breakdown=" + str(dict(counts))
    return msg
