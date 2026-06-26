"""
Ensemble detection head (XGBoost + RandomForest + LightGBM soft voting)
and attribution head (sklearn MLP).
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ensemble detector
# ---------------------------------------------------------------------------

def build_ensemble(cfg: dict, seed: int = 42) -> object:
    xgb = XGBClassifier(
        n_estimators=cfg["xgb_n_estimators"],
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=cfg["rf_n_estimators"],
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    lgb = LGBMClassifier(
        n_estimators=cfg["lgb_n_estimators"],
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    weights = cfg.get("weights", [0.4, 0.3, 0.3])
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("lgb", lgb)],
        voting="soft",
        weights=weights,
    )
    return ensemble


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    save_path: str = "outputs/models/detector.pkl",
    seed: int = 42,
) -> object:
    log.info("Training ensemble detector on %d samples …", len(X_train))
    clf = build_ensemble(cfg, seed)
    clf.fit(X_train, y_train)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f, protocol=4)
    log.info("Detector saved → %s", save_path)
    return clf


def predict_detector(clf, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (binary_predictions, fake_probabilities)."""
    proba = clf.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return preds, proba


# ---------------------------------------------------------------------------
# Attribution classifier
# ---------------------------------------------------------------------------

def train_attribution(
    X_train: np.ndarray,
    y_train_str: List[str],
    att_classes: List[str],
    cfg: dict,
    le: LabelEncoder,
    save_path: str = "outputs/models/attribution.pkl",
    seed: int = 42,
) -> Tuple[MLPClassifier, LabelEncoder]:
    """Train MLP attribution classifier on fused features of fake samples only."""
    # Use only fake samples (attribution is meaningless for real faces)
    X_fake = X_train
    y_idx  = le.transform(y_train_str)

    log.info("Training attribution MLP on %d fake samples, %d classes …",
             len(X_fake), len(att_classes))

    hidden = cfg.get("hidden_dim", 64)
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden, hidden),
        activation="relu",
        max_iter=cfg.get("epochs", 60) * 5,
        learning_rate_init=1e-3,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp.fit(X_fake, y_idx)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"mlp": mlp, "le": le}, f, protocol=4)
    log.info("Attribution model saved → %s", save_path)
    return mlp, le


def predict_attribution(
    mlp: MLPClassifier,
    le: LabelEncoder,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (class_name_predictions, probability_matrix)."""
    idx  = mlp.predict(X)
    prob = mlp.predict_proba(X)
    return le.inverse_transform(idx), prob
