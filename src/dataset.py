"""
Dataset Loading and Split Management for MIMII Acoustic Project
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

MACHINES = ["fan", "valve", "slider", "pump"]
CONDITIONS = ["-6_dB", "0_dB", "6_dB"]
DATASETS = [f"{c}_{m}" for m in MACHINES for c in CONDITIONS]


def load_project_data(project_dir: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads pre-extracted features and existing train/val/test split indices.
    Returns:
        X: np.ndarray [2400, 128, 157]
        y: np.ndarray [2400]
        metadata: pd.DataFrame [2400 rows]
        train_idx: np.ndarray [1680]
        val_idx: np.ndarray [360]
        test_idx: np.ndarray [360]
    """
    meta_path = os.path.join(project_dir, "all_metadata.csv")
    train_idx_path = os.path.join(project_dir, "train_idx.npy")
    val_idx_path = os.path.join(project_dir, "val_idx.npy")
    test_idx_path = os.path.join(project_dir, "test_idx.npy")

    if not all(os.path.exists(p) for p in [meta_path, train_idx_path, val_idx_path, test_idx_path]):
        raise FileNotFoundError(f"Missing required index/metadata files in {project_dir}")

    metadata = pd.read_csv(meta_path)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    X_parts, y_parts = [], []
    for d in DATASETS:
        feat_path = os.path.join(project_dir, f"{d}_features.npz")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file missing: {feat_path}")
        with np.load(feat_path) as data:
            X_parts.append(data["X"])
            y_parts.append(data["y"])

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)

    assert len(X) == len(y) == len(metadata), "Feature array length does not match metadata length!"
    return X, y, metadata, train_idx, val_idx, test_idx


def get_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders matching the notebook settings.
    Tensors are shaped as [batch_size, 1, 128, 157].
    """
    def make_loader(indices, shuffle=False):
        features = torch.from_numpy(X[indices]).float().unsqueeze(1)
        labels = torch.from_numpy(y[indices]).float()
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    return train_loader, val_loader, test_loader
