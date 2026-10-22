"""
Training Engine for Accuracy Improvement Experiments (v1)
Trains:
1. baseline_cnn: BaselineFanCNN on leak-free split
2. baseline_cnn_gru: BaselineFanCNNGRU on leak-free split
3. improved_cnn: ImprovedAcousticCNN on leak-free split with Z-score feature standardization
4. improved_cnn_gru: ImprovedAcousticCNNGRU on leak-free split with Z-score feature standardization

Features:
- AdamW optimizer with decoupled weight decay
- CosineAnnealingLR learning rate schedule
- Gradient clipping (max_norm = 1.0)
- Validation ROC-AUC monitored early stopping (patience = 6)
- Training & validation accuracy and loss tracking per epoch
- Held-out test set remains COMPLETELY UNTOUCHED during training!
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_DIR = r"C:\Capstone Project files"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

EXP_DIR = os.path.join(PROJECT_DIR, "experiments", "accuracy_improvement_v1")
CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")
HISTORY_DIR = os.path.join(EXP_DIR, "history")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
SPLITS_DIR = os.path.join(EXP_DIR, "splits")

for d in [CHECKPOINTS_DIR, HISTORY_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Set seed for strict deterministic reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")
print(f"Executing training engine on: {DEVICE}")

# 1. Load Data and Grouped Splits
from src.dataset import load_project_data
from experiments.accuracy_improvement_v1.models_v1 import (
    BaselineFanCNN, BaselineFanCNNGRU, ImprovedAcousticCNN, ImprovedAcousticCNNGRU
)

X, y, metadata, _, _, _ = load_project_data(PROJECT_DIR)

train_idx = np.load(os.path.join(SPLITS_DIR, "train_idx_grouped.npy"))
val_idx = np.load(os.path.join(SPLITS_DIR, "val_idx_grouped.npy"))
# Note: test_idx is intentionally NOT loaded here to guarantee zero contamination!

print(f"Data loaded: Train clips = {len(train_idx)}, Validation clips = {len(val_idx)}")

# Load training-fitted scaler parameters
scaler_data = np.load(os.path.join(SPLITS_DIR, "scaler_params.npz"))
train_mean = float(scaler_data["global_mean"])
train_std = float(scaler_data["global_std"])

# Standardized feature arrays (fitted on training data only)
X_std = (X - train_mean) / (train_std + 1e-6)

BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 6

def get_loaders(use_standardization=True):
    src_X = X_std if use_standardization else X
    
    train_x = torch.from_numpy(src_X[train_idx]).float().unsqueeze(1)
    train_y = torch.from_numpy(y[train_idx]).float()
    val_x = torch.from_numpy(src_X[val_idx]).float().unsqueeze(1)
    val_y = torch.from_numpy(y[val_idx]).float()

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_ld, val_ld


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE
):
    print("\n" + "=" * 65)
    print(f"TRAINING EXPERIMENT: {model_name.upper()}")
    print(f"Hyperparameters: LR={lr}, Weight Decay={weight_decay}, Max Epochs={max_epochs}, Patience={patience}")
    print("=" * 65)

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pt")

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_preds_list = []
        train_targets_list = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # Gradient clipping to prevent gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item() * len(yb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds_list.extend((probs >= 0.5).astype(int))
            train_targets_list.extend(yb.cpu().numpy().astype(int))

        train_loss = train_loss_total / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets_list, train_preds_list)

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_total += loss.item() * len(yb)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_list.extend(probs)
                val_targets_list.extend(yb.cpu().numpy().astype(int))

        val_loss = val_loss_total / len(val_loader.dataset)
        val_probs_arr = np.asarray(val_probs_list)
        val_targets_arr = np.asarray(val_targets_list)
        val_preds_arr = (val_probs_arr >= 0.5).astype(int)

        val_acc = accuracy_score(val_targets_arr, val_preds_arr)
        val_f1 = f1_score(val_targets_arr, val_preds_arr, zero_division=0)
        val_auc = roc_auc_score(val_targets_arr, val_probs_arr)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "lr": current_lr
        }
        history.append(record)

        # Log progress
        print(f"Epoch {epoch:02d}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")

        # Check early stopping condition (maximize validation ROC-AUC)
        if val_auc > (best_val_auc + 1e-4):
            best_val_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"   -> [BEST] Checkpoint saved at epoch {epoch} with Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no validation AUC improvement for {patience} epochs).")
                break

    elapsed = time.perf_counter() - start_time
    print(f"Training completed in {elapsed:.1f} seconds. Best Val AUC = {best_val_auc:.4f}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(HISTORY_DIR, f"{model_name}_history.csv")
    history_df.to_csv(history_path, index=False)

    # Plot training and validation loss & accuracy curves (Task 5: Overfitting Analysis)
    plot_learning_curves(history_df, model_name)

    return history_df, best_val_auc


def plot_learning_curves(df: pd.DataFrame, model_name: str):
    """
    Plots training vs validation loss and accuracy across epochs to analyze overfitting/underfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Loss curve
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2.2, marker="o", ms=4)
    axes[0].plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#d62728", lw=2.2, marker="s", ms=4)
    axes[0].set_title(f"{model_name.upper()}: Loss Progression", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Binary Cross Entropy Loss", fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    axes[1].plot(df["epoch"], df["train_acc"] * 100, label="Train Accuracy", color="#1f77b4", lw=2.2, marker="o", ms=4)
    axes[1].plot(df["epoch"], df["val_acc"] * 100, label="Validation Accuracy", color="#2ca02c", lw=2.2, marker="^", ms=4)
    axes[1].plot(df["epoch"], df["val_auc"] * 100, label="Validation ROC-AUC", color="#ff7f0e", lw=2.0, linestyle="--")
    axes[1].set_title(f"{model_name.upper()}: Accuracy & ROC-AUC Progression", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Percentage (%)", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, f"{model_name}_learning_curves.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to: {plot_path}")


# ------------------------------------------------------------------------------
# Run Documented Experiment Suite
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader_std, val_loader_std = get_loaders(use_standardization=True)
    train_loader_raw, val_loader_raw = get_loaders(use_standardization=False)

    experiments = [
        {
            "name": "baseline_cnn",
            "model_fn": BaselineFanCNN,
            "loader": (train_loader_raw, val_loader_raw),
            "lr": 3e-4,
            "weight_decay": 1e-4
        },
        {
            "name": "baseline_cnn_gru",
            "model_fn": BaselineFanCNNGRU,
            "loader": (train_loader_raw, val_loader_raw),
            "lr": 3e-4,
            "weight_decay": 1e-4
        },
        {
            "name": "improved_cnn",
            "model_fn": ImprovedAcousticCNN,
            "loader": (train_loader_std, val_loader_std),
            "lr": 3e-4,
            "weight_decay": 1e-3
        },
        {
            "name": "improved_cnn_gru",
            "model_fn": ImprovedAcousticCNNGRU,
            "loader": (train_loader_std, val_loader_std),
            "lr": 3e-4,
            "weight_decay": 1e-3
        }
    ]

    summary_results = []
    for exp in experiments:
        m = exp["model_fn"]()
        tr_ld, vl_ld = exp["loader"]
        history, best_auc = train_single_model(
            model=m,
            model_name=exp["name"],
            train_loader=tr_ld,
            val_loader=vl_ld,
            lr=exp["lr"],
            weight_decay=exp["weight_decay"]
        )
        best_epoch_row = history.loc[history["val_auc"].idxmax()]
        summary_results.append({
            "experiment": exp["name"],
            "best_epoch": int(best_epoch_row["epoch"]),
            "train_loss": best_epoch_row["train_loss"],
            "train_acc": best_epoch_row["train_acc"],
            "val_loss": best_epoch_row["val_loss"],
            "val_acc": best_epoch_row["val_acc"],
            "val_f1": best_epoch_row["val_f1"],
            "val_auc": best_epoch_row["val_auc"]
        })

    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(EXP_DIR, "validation_model_selection_table.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 65)
    print("VALIDATION MODEL SELECTION TABLE (Best Models Selected by Val AUC)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
