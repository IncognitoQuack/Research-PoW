"""
partition.py
-------------
Splits the processed, real CVE dataset (config.PROCESSED_CSV) into:
  1. A held-out global test set (stratified by severity label), used to
     evaluate utility identically across all three conditions, and to
     supply the "non-member" half of the membership-inference attack.
  2. NUM_CLIENTS non-overlapping IID training shards, simulating
     NUM_CLIENTS organizations each holding a private slice of
     vulnerability-intelligence records that never leaves their own
     machine in plaintext form during federated training.

IID sharding is used as the main-paper configuration for a clean,
interpretable utility/privacy/overhead comparison; a non-IID split
(e.g. by publication year or CWE family) is noted as future work in the
Limitations section rather than silently assumed away.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def make_splits(processed_csv: str = C.PROCESSED_CSV, out_dir: str = C.SPLITS_DIR):
    df = pd.read_csv(processed_csv)
    assert len(df) > 0, "processed_csv is empty -- run data_fetch.py first"

    train_df, test_df = train_test_split(
        df, test_size=C.TEST_FRACTION, random_state=C.SEED, stratify=df["severity_label"]
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    test_df.to_csv(Path(out_dir) / "test.csv", index=False)

    rng = np.random.default_rng(C.SEED)
    shuffled_idx = rng.permutation(len(train_df))
    shard_indices = np.array_split(shuffled_idx, C.NUM_CLIENTS)

    train_df_reset = train_df.reset_index(drop=True)
    client_sizes = []
    for i, idx in enumerate(shard_indices):
        shard = train_df_reset.iloc[idx]
        shard.to_csv(Path(out_dir) / f"client_{i}.csv", index=False)
        client_sizes.append(len(shard))

    print(f"[partition] total records: {len(df)}  train: {len(train_df)}  test: {len(test_df)}")
    print(f"[partition] {C.NUM_CLIENTS} client shards, sizes: {client_sizes}")
    print(f"[partition] test-set class balance:\n{test_df['severity_label'].value_counts()}")
    return train_df_reset, test_df


def load_client(client_id: int, splits_dir: str = C.SPLITS_DIR):
    path = Path(splits_dir) / f"client_{client_id}.csv"
    df = pd.read_csv(path)
    return df["description"].tolist(), [C.LABEL2ID[l] for l in df["severity_label"]]


def load_test(splits_dir: str = C.SPLITS_DIR):
    df = pd.read_csv(Path(splits_dir) / "test.csv")
    return df["description"].tolist(), [C.LABEL2ID[l] for l in df["severity_label"]]


def load_all_train_pooled(splits_dir: str = C.SPLITS_DIR, num_clients: int = C.NUM_CLIENTS):
    """Used only by the centralized baseline, which is allowed to pool everyone's data."""
    texts, labels = [], []
    for i in range(num_clients):
        t, l = load_client(i, splits_dir)
        texts.extend(t)
        labels.extend(l)
    return texts, labels


if __name__ == "__main__":
    make_splits()
