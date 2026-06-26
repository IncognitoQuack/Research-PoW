"""
Demo dataset builder — uses Labeled Faces in the Wild (LFW) as real faces
and applies controlled frequency-domain perturbations to simulate deepfake
artifacts in the DCT domain.

This is for PIPELINE TESTING ONLY.  Paper results use FF++.

Usage:
  python scripts/demo_data.py --output_root data --n 300

Requirements: scikit-learn (fetch_lfw_people), Pillow, numpy
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm


def _save(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(str(path), quality=95)


def _make_fake(img_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate GAN-like frequency artifacts:
      1. Add structured high-frequency noise in DCT domain.
      2. Apply slight colour shift (GAN colour bias).
    This is a controlled proxy — NOT a real deepfake.
    """
    import cv2

    img = img_arr.astype(np.float32)
    # Per-channel DCT perturbation
    for ch in range(img.shape[2] if img.ndim == 3 else 1):
        plane = img[:, :, ch] if img.ndim == 3 else img
        d = cv2.dct(plane)
        # Inject low-magnitude structured noise at mid-frequencies
        noise = rng.uniform(-8, 8, d.shape).astype(np.float32)
        mask = np.zeros_like(d)
        mask[4:20, 4:20] = 1.0          # mid-frequency band
        d += noise * mask
        restored = cv2.idct(d)
        if img.ndim == 3:
            img[:, :, ch] = restored
        else:
            img = restored

    # Colour shift
    if img.ndim == 3:
        img += rng.uniform(-12, 12, (1, 1, 3)).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--output_root", default="data")
    pa.add_argument("--n", type=int, default=300,
                    help="Number of real face images to download (fake = same count)")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_root)

    print("Downloading LFW people dataset …")
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True)
    imgs = lfw.images                          # (N, H, W, 3)
    n_total = min(args.n, len(imgs))
    idx = rng.permutation(len(imgs))[:n_total]
    imgs = imgs[idx]

    # Split 70 / 15 / 15
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)

    splits = {
        "train": imgs[:n_train],
        "val":   imgs[n_train : n_train + n_val],
        "test":  imgs[n_train + n_val :],
    }

    for split, split_imgs in splits.items():
        real_dir = out / split / "real"
        fake_dir = out / split / "fake" / "SimulatedDeepfake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(tqdm(split_imgs, desc=f"  {split}")):
            img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
            img_resized = np.array(
                Image.fromarray(img_u8).resize((224, 224), Image.BILINEAR)
            )
            _save(img_resized, real_dir / f"real_{i:04d}.jpg")
            fake = _make_fake(img_resized.copy(), rng)
            _save(fake, fake_dir / f"fake_{i:04d}.jpg")

    print("\nDemo dataset ready:")
    for split in ("train", "val", "test"):
        r = len(list((out / split / "real").glob("*.jpg")))
        f = len(list((out / split / "fake" / "SimulatedDeepfake").glob("*.jpg")))
        print(f"  {split:<6}  real={r}  fake={f}")
    print("\nNOTE: These are simulated artifacts. Use FF++ for paper-quality results.")


if __name__ == "__main__":
    main()
