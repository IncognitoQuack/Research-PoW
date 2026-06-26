"""
FF++ frame extractor.

Expected FF++ directory layout (after downloading via the official script):
  <ffpp_root>/
    original_sequences/actors/c23/videos/  (1000 MP4s, real faces)
    manipulated_sequences/
      Deepfakes/c23/videos/
      Face2Face/c23/videos/
      FaceSwap/c23/videos/
      NeuralTextures/c23/videos/

Usage:
  python scripts/prepare_ffpp.py \
      --ffpp_root /path/to/FaceForensics++ \
      --output_root data \
      --compression c23 \
      --frames_per_video 10 \
      --max_videos 150

Extracted frames are saved as:
  data/train/real/   data/val/real/   data/test/real/
  data/train/fake/DeepFakes/   data/train/fake/Face2Face/  …
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Official 720/140/140 split JSON files (present in the FF++ download)
SPLIT_FILES = {
    "train": "splits/train.json",
    "val":   "splits/val.json",
    "test":  "splits/test.json",
}


def extract_frames(video_path: Path, out_dir: Path, n_frames: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    step = max(1, total // n_frames)
    saved = 0
    frame_idx = 0
    while saved < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        fname = out_dir / f"{video_path.stem}_f{frame_idx:04d}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        frame_idx += step

    cap.release()
    return saved


def load_split_ids(ffpp_root: Path, split_name: str):
    split_file = ffpp_root / SPLIT_FILES[split_name]
    if not split_file.exists():
        # Fallback: read all video IDs and divide 72/14/14
        all_ids = sorted(
            p.stem for p in (ffpp_root / "original_sequences" / "actors" / "c23" / "videos").glob("*.mp4")
        )
        n = len(all_ids)
        splits = {
            "train": all_ids[:int(0.72 * n)],
            "val":   all_ids[int(0.72 * n):int(0.86 * n)],
            "test":  all_ids[int(0.86 * n):],
        }
        return splits[split_name]
    with open(split_file) as f:
        pairs = json.load(f)
    ids = set()
    for pair in pairs:
        ids.update(pair)
    return sorted(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ffpp_root",     required=True,  help="Root of FF++ download")
    p.add_argument("--output_root",   default="data", help="Where to write frames")
    p.add_argument("--compression",   default="c23",  choices=["c23", "c40"])
    p.add_argument("--frames_per_video", type=int, default=10)
    p.add_argument("--max_videos",    type=int, default=150,
                   help="Cap per class per split (keep runtime manageable)")
    args = p.parse_args()

    ffpp = Path(args.ffpp_root)
    out  = Path(args.output_root)

    for split in ("train", "val", "test"):
        ids = load_split_ids(ffpp, split)[:args.max_videos]
        print(f"\n[{split}]  {len(ids)} video IDs")

        # Real
        real_vid_dir = ffpp / "original_sequences" / "actors" / args.compression / "videos"
        for vid_id in tqdm(ids, desc=f"  real/{split}"):
            vid = real_vid_dir / f"{vid_id}.mp4"
            if not vid.exists():
                continue
            extract_frames(vid, out / split / "real", args.frames_per_video)

        # Fake (each manipulation type)
        for manip in MANIPULATIONS:
            manip_vid_dir = (
                ffpp / "manipulated_sequences" / manip / args.compression / "videos"
            )
            if not manip_vid_dir.exists():
                print(f"  Warning: {manip_vid_dir} not found, skipping.")
                continue
            for vid_id in tqdm(ids, desc=f"  fake/{manip}/{split}"):
                # FF++ fake videos are named <src_id>_<tgt_id>.mp4
                matches = list(manip_vid_dir.glob(f"{vid_id}_*.mp4")) + \
                          list(manip_vid_dir.glob(f"*_{vid_id}.mp4"))
                for vid in matches[:1]:
                    extract_frames(vid, out / split / "fake" / manip, args.frames_per_video)

    print("\nDone.  Frame counts:")
    for split in ("train", "val", "test"):
        real_count = len(list((out / split / "real").glob("*.jpg")))
        fake_count = sum(
            len(list((out / split / "fake" / m).glob("*.jpg")))
            for m in MANIPULATIONS
            if (out / split / "fake" / m).exists()
        )
        print(f"  {split:<6}  real={real_count}  fake={fake_count}")


if __name__ == "__main__":
    main()
