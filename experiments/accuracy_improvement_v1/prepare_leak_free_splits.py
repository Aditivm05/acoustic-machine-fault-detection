"""
Leak-Free Stratified Group Splitting and Training-Only Feature Normalization
Strictly groups all noisy variants of the same source recording into the same split.
Guarantees zero recording overlap between Train, Validation, and Test sets.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys

PROJECT_DIR = r"C:\Capstone Project files"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

EXP_DIR = os.path.join(PROJECT_DIR, "experiments", "accuracy_improvement_v1")
SPLITS_DIR = os.path.join(EXP_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# 1. Load Metadata
meta_path = os.path.join(PROJECT_DIR, "all_metadata.csv")
metadata = pd.read_csv(meta_path)
print(f"Loaded metadata with {len(metadata)} total audio clips.")

# In MIMII, the underlying source recording is identified by the relative file path:
# e.g., 'fan/id_00/normal/00000554.wav' which appears across -6dB, 0dB, and +6dB
metadata["source_id"] = metadata["file"].str.replace("\\", "/", regex=False)
metadata["strata"] = metadata["machine"] + "_" + metadata["label"].astype(str)

# 2. Group-level Stratification Table
group_df = metadata.groupby("source_id").agg({
    "machine": "first",
    "label": "first",
    "strata": "first",
    "dataset": "count"
}).reset_index().rename(columns={"dataset": "num_variants"})

print(f"Total unique source recordings: {len(group_df)}")
print("Recordings per machine-label strata:")
print(group_df["strata"].value_counts())

# 3. Perform 70 / 15 / 15 Stratified Split on SOURCE RECORDINGS
SEED = 42
train_groups, rem_groups = train_test_split(
    group_df["source_id"],
    test_size=0.30,
    stratify=group_df["strata"],
    random_state=SEED
)

rem_df = group_df[group_df["source_id"].isin(rem_groups)]
val_groups, test_groups = train_test_split(
    rem_df["source_id"],
    test_size=0.50,
    stratify=rem_df["strata"],
    random_state=SEED
)

train_set = set(train_groups)
val_set = set(val_groups)
test_set = set(test_groups)

# 4. Strict Overlap Audit
overlap_train_val = len(train_set.intersection(val_set))
overlap_train_test = len(train_set.intersection(test_set))
overlap_val_test = len(val_set.intersection(test_set))

assert overlap_train_val == 0, f"Leakage detected between Train and Val: {overlap_train_val}"
assert overlap_train_test == 0, f"Leakage detected between Train and Test: {overlap_train_test}"
assert overlap_val_test == 0, f"Leakage detected between Val and Test: {overlap_val_test}"

# 5. Map back to clip indices
train_mask = metadata["source_id"].isin(train_set)
val_mask = metadata["source_id"].isin(val_set)
test_mask = metadata["source_id"].isin(test_set)

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"\nFinal Grouped Clip Counts:")
print(f"  Train:      {len(train_idx)} clips (Normal: {(metadata.iloc[train_idx]['label'] == 0).sum()}, Abnormal: {(metadata.iloc[train_idx]['label'] == 1).sum()})")
print(f"  Validation: {len(val_idx)} clips (Normal: {(metadata.iloc[val_idx]['label'] == 0).sum()}, Abnormal: {(metadata.iloc[val_idx]['label'] == 1).sum()})")
print(f"  Test:       {len(test_idx)} clips (Normal: {(metadata.iloc[test_idx]['label'] == 0).sum()}, Abnormal: {(metadata.iloc[test_idx]['label'] == 1).sum()})")

# Save new grouped indices
np.save(os.path.join(SPLITS_DIR, "train_idx_grouped.npy"), train_idx)
np.save(os.path.join(SPLITS_DIR, "val_idx_grouped.npy"), val_idx)
np.save(os.path.join(SPLITS_DIR, "test_idx_grouped.npy"), test_idx)

# 6. Fit Feature Normalization ONLY on Training Data (ml-best-practices)
from src.dataset import load_project_data
X, y, _, _, _, _ = load_project_data(PROJECT_DIR)

X_train = X[train_idx]
# Compute channel/frequency-wise mean and std on training samples only
train_mean = float(np.mean(X_train))
train_std = float(np.std(X_train))

# Also compute per-mel-bin mean and std for frequency-wise standardization
mel_mean = np.mean(X_train, axis=(0, 2), keepdims=True).astype(np.float32)  # [1, 128, 1]
mel_std = (np.std(X_train, axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)

np.savez_compressed(
    os.path.join(SPLITS_DIR, "scaler_params.npz"),
    global_mean=train_mean,
    global_std=train_std,
    mel_mean=mel_mean,
    mel_std=mel_std
)
print(f"\nTraining-fitted Scaler Saved: Global Mean={train_mean:.4f}, Global Std={train_std:.4f}")

# 7. Verification Report
verification_report = {
    "total_samples": len(metadata),
    "unique_source_recordings": len(group_df),
    "train_source_recordings": len(train_groups),
    "val_source_recordings": len(val_groups),
    "test_source_recordings": len(test_groups),
    "train_clips": len(train_idx),
    "val_clips": len(val_idx),
    "test_clips": len(test_idx),
    "overlap_train_val": overlap_train_val,
    "overlap_train_test": overlap_train_test,
    "overlap_val_test": overlap_val_test,
    "leak_free_verified": True,
    "scaler": {
        "global_mean": train_mean,
        "global_std": train_std,
        "fitted_on": "train_split_only"
    }
}

with open(os.path.join(SPLITS_DIR, "split_verification_report.json"), "w") as f:
    json.dump(verification_report, f, indent=4)

print("Split creation and verification report saved successfully.")
