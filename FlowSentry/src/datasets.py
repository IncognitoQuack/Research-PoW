"""
datasets.py
===========
Wraps preprocessed feature/label arrays into PyTorch Datasets.

Two views of the same tabular flow features are produced per sample:
  - a 1-D feature vector (consumed by the CNN branch as a single "signal")
  - a short temporal window of the W most recent flows for the same sample
    stream (consumed by the RNN branch), built by a simple sliding window
    over the (already shuffled-free) sequential order of the processed
    array. This mirrors how a real-time NIDS would buffer the last W flows
    before scoring the current one.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedFlowDataset(Dataset):
    """Builds (window, current_vector, label) triples.

    Args:
        X: (N, F) float32 array of preprocessed flow features, in temporal order.
        y: (N,) int64 array of labels.
        window_length: number of preceding flows (including current) fed to the RNN.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_length: int = 10):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.w = window_length
        self.n_features = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        start = max(0, idx - self.w + 1)
        window = self.X[start: idx + 1]
        if len(window) < self.w:
            pad = np.zeros((self.w - len(window), self.n_features), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        current = self.X[idx]
        label = self.y[idx]
        return (
            torch.from_numpy(window),          # (W, F) for the RNN branch
            torch.from_numpy(current),          # (F,)   for the CNN branch
            torch.tensor(label, dtype=torch.long),
        )


def make_loaders(X_train, y_train, X_test, y_test, window_length, batch_size,
                  val_fraction: float = 0.1, seed: int = 42, num_workers: int = 0):
    from torch.utils.data import DataLoader, Subset

    n_train = len(X_train)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    n_val = int(val_fraction * n_train)
    val_idx = np.sort(idx[:n_val])          # keep temporal order within split
    train_idx = np.sort(idx[n_val:])

    full_train_ds = WindowedFlowDataset(X_train, y_train, window_length)
    test_ds = WindowedFlowDataset(X_test, y_test, window_length)

    train_ds = Subset(full_train_ds, train_idx.tolist())
    val_ds = Subset(full_train_ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader
