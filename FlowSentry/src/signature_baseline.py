"""
signature_baseline.py
======================
Kept in its own module (rather than inside baselines.py's __main__ script)
so that joblib/pickle can locate the class consistently when the fitted
object is saved by baselines.py and later loaded by evaluate.py or
latency_bench.py — pickle records classes by "module.ClassName", so a class
defined in a script run as __main__ cannot be unpickled from a different
entry-point script.
"""
import numpy as np


class SignatureRuleBaseline:
    """A transparent, fixed-threshold rule applied to one feature dimension —
    a simplistic, honest analogue of a static signature rule, used purely
    as a latency/behaviour reference point (signature systems are not
    expected to be the accuracy leader in this comparison).
    """

    def __init__(self):
        self.feature_idx_ = None
        self.threshold_ = None
        self.direction_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        normal_mask = y_train == 0
        attack_mask = y_train == 1
        if normal_mask.sum() == 0 or attack_mask.sum() == 0:
            self.feature_idx_ = 0
            self.threshold_ = 0.0
            self.direction_ = 1
            return self
        mean_diff = X_train[attack_mask].mean(axis=0) - X_train[normal_mask].mean(axis=0)
        self.feature_idx_ = int(np.argmax(np.abs(mean_diff)))
        self.direction_ = 1 if mean_diff[self.feature_idx_] >= 0 else -1
        normal_vals = X_train[normal_mask, self.feature_idx_]
        attack_vals = X_train[attack_mask, self.feature_idx_]
        self.threshold_ = float((normal_vals.mean() + attack_vals.mean()) / 2.0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        vals = X[:, self.feature_idx_] * self.direction_
        thr = self.threshold_ * self.direction_
        return (vals > thr).astype(np.int64)
