"""
data_utils.py
=============
Loading, cleaning, encoding and scaling for NSL-KDD and CIC-IDS2017.

This module deliberately does NOT ship any dataset. It expects the raw
files to already be downloaded by the user (see README §2) and writes a
cleaned/encoded cache to `data/processed/<dataset>/` so that later steps
(train.py, baselines.py) load quickly and deterministically.

Usage:
    python3 src/data_utils.py --dataset nsl_kdd --raw_dir data/raw/nsl_kdd
    python3 src/data_utils.py --dataset cic_ids2017 --raw_dir data/raw/cic_ids2017
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

RNG_SEED = 42

# ---------------------------------------------------------------------------
# NSL-KDD schema (41 features + label + difficulty), standard published names
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
    "difficulty",
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# Mapping from raw NSL-KDD attack labels to the four coarse attack categories
# used widely in the literature (Tavallaee et al., 2009 taxonomy).
NSL_KDD_ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "processtable": "dos",
    "udpstorm": "dos", "apache2": "dos", "worm": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l",
    "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l",
    "snmpgetattack": "r2l", "httptunnel": "r2l", "sendmail": "r2l",
    "named": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def _clean_label(raw_label: str) -> str:
    return raw_label.strip().lower().replace(".", "")


def load_nsl_kdd_raw(raw_dir: str) -> pd.DataFrame:
    train_path = os.path.join(raw_dir, "KDDTrain+.txt")
    test_path = os.path.join(raw_dir, "KDDTest+.txt")
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected NSL-KDD file not found: {p}\n"
                "Download from https://www.unb.ca/cic/datasets/nsl.html "
                "and place KDDTrain+.txt / KDDTest+.txt in this folder."
            )
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    train_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.drop(columns=["difficulty"])
    df["label"] = df["label"].apply(_clean_label)
    df["attack_category"] = df["label"].map(NSL_KDD_ATTACK_MAP).fillna("unknown")
    return df


def load_cic_ids2017_raw(raw_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            "Download the MachineLearningCSV bundle from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and place the "
            "per-day CSVs in this folder."
        )
    frames = []
    for f in csv_files:
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in CIC-IDS2017 CSVs.")
    df = df.rename(columns={"Label": "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["attack_category"] = np.where(df["label"] == "benign", "normal", "attack")
    return df


def _clean_numeric(df: pd.DataFrame, exclude_cols) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in exclude_cols]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    return df


def preprocess(dataset: str, raw_dir: str, out_dir: str, binary_labels: bool = True,
               scaler_name: str = "standard", seed: int = RNG_SEED,
               max_train_samples: int = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    if dataset == "nsl_kdd":
        df = load_nsl_kdd_raw(raw_dir)
        cat_cols = NSL_KDD_CATEGORICAL
        exclude = cat_cols + ["label", "attack_category", "split"]
        df = _clean_numeric(df, exclude_cols=exclude)
        for c in cat_cols:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"
        feature_cols = [c for c in df.columns
                         if c not in ("label", "attack_category", "split")]
    elif dataset == "cic_ids2017":
        df = load_cic_ids2017_raw(raw_dir)
        exclude = ["label", "attack_category"]
        df = _clean_numeric(df, exclude_cols=exclude)
        feature_cols = [c for c in df.columns
                         if c not in ("label", "attack_category")]
        # random stratified 80/20 split since CIC-IDS2017 ships unsplit
        idx = np.arange(len(df))
        rng.shuffle(idx)
        split_point = int(0.8 * len(df))
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[idx[:split_point]] = True
        test_mask = ~train_mask
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y_source = "attack_category" if binary_labels else "label"
    label_encoder = LabelEncoder()
    if dataset == "nsl_kdd" and binary_labels:
        y_all = (df["attack_category"] != "normal").astype(int).values
        class_names = ["normal", "attack"]
    elif dataset == "cic_ids2017" and binary_labels:
        y_all = (df["attack_category"] != "normal").astype(int).values
        class_names = ["normal", "attack"]
    else:
        y_all = label_encoder.fit_transform(df[y_source].astype(str))
        class_names = list(label_encoder.classes_)

    X_all = df[feature_cols].values.astype(np.float32)

    X_train, X_test = X_all[train_mask], X_all[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]

    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    original_n_train = int(X_train.shape[0])
    subsampled = False
    if max_train_samples is not None and original_n_train > max_train_samples:
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=max_train_samples,
            stratify=y_train,
            random_state=seed,
        )
        subsampled = True

    np.save(os.path.join(out_dir, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(out_dir, "X_test.npy"), X_test.astype(np.float32))
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(out_dir, "y_test.npy"), y_test.astype(np.int64))

    meta = {
        "dataset": dataset,
        "n_features": X_train.shape[1],
        "feature_cols": feature_cols,
        "class_names": class_names,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "original_n_train": original_n_train,
        "subsampled_train": subsampled,
        "max_train_samples": max_train_samples,
        "binary_labels": binary_labels,
        "scaler": scaler_name,
        "seed": seed,
        "class_balance_train": {
            str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{dataset}] processed -> {out_dir}")
    print(json.dumps(meta, indent=2))


def load_processed(dataset: str, processed_root: str = "data/processed"):
    d = os.path.join(processed_root, dataset)
    X_train = np.load(os.path.join(d, "X_train.npy"))
    X_test = np.load(os.path.join(d, "X_test.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return X_train, X_test, y_train, y_test, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "cic_ids2017"])
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_root", default="data/processed")
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--multiclass", dest="binary", action="store_false")
    parser.add_argument("--scaler", default="standard", choices=["standard", "minmax"])
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--max_train_samples", type=int, default=None,
                         help="Optional cap on training-set size via stratified "
                              "subsampling (test set is never subsampled). Useful "
                              "for large datasets like CIC-IDS2017 on laptop hardware.")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    preprocess(args.dataset, args.raw_dir, out_dir,
               binary_labels=args.binary, scaler_name=args.scaler, seed=args.seed,
               max_train_samples=args.max_train_samples)
