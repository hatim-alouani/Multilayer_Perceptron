"""Dataset loading and preprocessing for the WDBC dataset.

The raw csv has no header and 32 columns:
    col 0      -> sample id      (NOT a feature, must be dropped)
    col 1      -> diagnosis      'M' (malignant) or 'B' (benign)
    col 2..31  -> 30 real-valued features
"""

import numpy as np
import pandas as pd

N_FEATURES = 30
CLASSES = ("B", "M")  # index 0 = benign, index 1 = malignant


def load_raw(path):
    """Read the raw csv exactly as it is on disk (no header)."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != N_FEATURES + 2:
        raise ValueError(
            f"expected {N_FEATURES + 2} columns in {path}, got {df.shape[1]}"
        )
    return df


def features_labels(df):
    """Split a raw dataframe into X (n_samples, 30) and y (n_samples,) of 'M'/'B'.

    Column 0 (the id) is dropped here. Leaving it in is one of the classic
    ways to make this project silently not work.
    """
    y = df.iloc[:, 1].to_numpy(dtype=str)
    X = df.iloc[:, 2:].to_numpy(dtype=np.float64)
    return X, y


def stratified_indices(y, train_size=0.8, seed=42):
    """Return (train_idx, valid_idx) preserving the class ratio in both parts.

    A plain random split can leave a small validation set skewed, which makes
    the validation curve noisy and hard to read.
    """
    rng = np.random.default_rng(seed)
    train_parts, valid_parts = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_size))
        train_parts.append(idx[:cut])
        valid_parts.append(idx[cut:])
    train_idx = np.concatenate(train_parts)
    valid_idx = np.concatenate(valid_parts)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def fit_standardizer(X):
    """Compute per-feature mean and std on the TRAINING set only.

    Returns (mean, std), each of shape (n_features,).
    Using statistics computed over the whole dataset is data leakage.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)  # a constant feature would divide by 0
    return mean, std


def standardize(X, mean, std):
    """Apply z-score scaling: (x - mean) / std."""
    return (X - mean) / std


def one_hot(y):
    """Encode labels as (n_samples, 2): B -> [1, 0], M -> [0, 1]."""
    out = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    for i, label in enumerate(CLASSES):
        out[y == label, i] = 1.0
    unknown = ~np.isin(y, CLASSES)
    if unknown.any():
        raise ValueError(f"unknown label(s) in dataset: {set(y[unknown])}")
    return out


def to_network_layout(X, Y=None):
    """Transpose from row-major tabular form to the column-major form the
    network uses internally.

        X: (n_samples, n_features) -> (n_features, n_samples)
        Y: (n_samples, n_classes)  -> (n_classes,  n_samples)

    Every matrix inside network.py is "one column per sample". Keeping that
    convention in one place is the easiest way to avoid shape bugs.
    """
    if Y is None:
        return X.T
    return X.T, Y.T
