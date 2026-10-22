"""
Comprehensive Evaluation and Comparison Engine for Accuracy Improvement Experiments (v1)
Evaluates:
- BaselineFanCNN vs ImprovedAcousticCNN
- BaselineFanCNNGRU vs ImprovedAcousticCNNGRU
on the leak-free grouped split.

Computes:
- Train, Validation, and Test Accuracy
- Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrices, FPR (False Positive Rate), FNR (False Negative Rate)
- Accuracy and F1 breakdown by Noise Condition (-6dB, 0dB, +6dB)
- Accuracy and F1 breakdown by Machine Type (Fan, Pump, Slider, Valve)
- Comparison against the previous random-split baseline
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

PROJECT_DIR = r"C:\Capstone Project files"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

EXP_DIR = os.path.join(PROJECT_DIR, "experiments", "accuracy_improvement_v1")
CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")
METRICS_DIR = os.path.join(EXP_DIR, "metrics")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
SPLITS_DIR = os.path.join(EXP_DIR, "splits")
CONFIGS_DIR = os.path.join(EXP_DIR, "configs")

for d in [METRICS_DIR, FIGURES_DIR, CONFIGS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cpu")

# 1. Load Data, Grouped Splits, and Scaler
from src.dataset import load_project_data
from experiments.accuracy_improvement_v1.models_v1 import (
    BaselineFanCNN, BaselineFanCNNGRU, ImprovedAcousticCNN, ImprovedAcousticCNNGRU
)

X, y, metadata, _, _, _ = load_project_data(PROJECT_DIR)

train_idx = np.load(os.path.join(SPLITS_DIR, "train_idx_grouped.npy"))
val_idx = np.load(os.path.join(SPLITS_DIR, "val_idx_grouped.npy"))
test_idx = np.load(os.path.join(SPLITS_DIR, "test_idx_grouped.npy"))

scaler_data = np.load(os.path.join(SPLITS_DIR, "scaler_params.npz"))
train_mean = float(scaler_data["global_mean"])
train_std = float(scaler_data["global_std"])

# Standardized feature arrays using training-fitted parameters
X_std = (X - train_mean) / (train_std + 1e-6)

print(f"Loaded {len(X)} clips across Train ({len(train_idx)}), Val ({len(val_idx)}), and Test ({len(test_idx)}).")


def predict_model(model, features, batch_size=32):
    model.eval()
    model.to(DEVICE)
    tensor_data = torch.from_numpy(features).float().unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    probs_list = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs_list.extend(probs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_sample = elapsed_ms / max(len(features), 1)
    return np.asarray(probs_list, dtype=np.float32), latency_per_sample


def calc_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tot_norm = tn + fp
    tot_abnorm = tp + fn
    fpr = float(fp / tot_norm) if tot_norm > 0 else 0.0
    fnr = float(fn / tot_abnorm) if tot_abnorm > 0 else 0.0

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "samples": len(y_true),
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "false_alarm_rate_fpr": fpr,
        "missed_fault_rate_fnr": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


# Define model catalog
model_catalog = [
    {
        "id": "baseline_cnn",
        "display_name": "Baseline CNN (Leak-Free)",
        "model_cls": BaselineFanCNN,
        "ckpt": os.path.join(CHECKPOINTS_DIR, "baseline_cnn_best.pt"),
        "use_std": False
    },
    {
        "id": "baseline_cnn_gru",
        "display_name": "Baseline CNN-GRU (Leak-Free)",
        "model_cls": BaselineFanCNNGRU,
        "ckpt": os.path.join(CHECKPOINTS_DIR, "baseline_cnn_gru_best.pt"),
        "use_std": False
    },
    {
        "id": "improved_cnn",
        "display_name": "Improved Acoustic CNN",
        "model_cls": ImprovedAcousticCNN,
        "ckpt": os.path.join(CHECKPOINTS_DIR, "improved_cnn_best.pt"),
        "use_std": True
    },
    {
        "id": "improved_cnn_gru",
        "display_name": "Improved Acoustic CNN-GRU",
        "model_cls": ImprovedAcousticCNNGRU,
        "ckpt": os.path.join(CHECKPOINTS_DIR, "improved_cnn_gru_best.pt"),
        "use_std": True
    }
]


def run_full_evaluation():
    print("=" * 65)
    print("EVALUATING EXPERIMENT MODELS ON UNTOUCHED TEST SET")
    print("=" * 65)

    all_split_metrics = []
    comparison_records = []
    model_predictions = {}

    for item in model_catalog:
        mid = item["id"]
        mname = item["display_name"]
        print(f"\nEvaluating: {mname}")

        model = item["model_cls"]()
        model.load_state_dict(torch.load(item["ckpt"], map_location=DEVICE))
        src_data = X_std if item["use_std"] else X

        model_predictions[mid] = {}

        # 1. Evaluate Train, Val, and Test
        for split_label, s_idx in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
            sub_x = src_data[s_idx]
            sub_y = y[s_idx]

            probs, lat = predict_model(model, sub_x)
            model_predictions[mid][split_label] = (probs, sub_y)

            m = calc_metrics(sub_y, probs, threshold=0.5)
            m["model_id"] = mid
            m["model_name"] = mname
            m["split"] = split_label
            m["latency_ms"] = lat
            all_split_metrics.append(m)

        # Extract test record for headline table
        tr_m = [r for r in all_split_metrics if r["model_id"] == mid and r["split"] == "Train"][0]
        vl_m = [r for r in all_split_metrics if r["model_id"] == mid and r["split"] == "Validation"][0]
        ts_m = [r for r in all_split_metrics if r["model_id"] == mid and r["split"] == "Test"][0]

        comparison_records.append({
            "model_id": mid,
            "architecture": mname,
            "train_accuracy": tr_m["accuracy"],
            "val_accuracy": vl_m["accuracy"],
            "test_accuracy": ts_m["accuracy"],
            "test_precision": ts_m["precision"],
            "test_recall": ts_m["recall"],
            "test_f1": ts_m["f1"],
            "test_roc_auc": ts_m["roc_auc"],
            "test_fpr": ts_m["false_alarm_rate_fpr"],
            "test_fnr": ts_m["missed_fault_rate_fnr"],
            "mean_latency_ms": ts_m["latency_ms"]
        })

    # Save metrics
    full_metrics_df = pd.DataFrame(all_split_metrics)
    full_metrics_path = os.path.join(METRICS_DIR, "full_diagnostic_metrics.csv")
    full_metrics_df.to_csv(full_metrics_path, index=False)

    comp_df = pd.DataFrame(comparison_records)
    comp_path = os.path.join(METRICS_DIR, "experiment_comparison_table.csv")
    comp_df.to_csv(comp_path, index=False)

    print("\n" + "=" * 65)
    print("EXPERIMENT HEADLINE COMPARISON TABLE (Test Set, Threshold = 0.50)")
    print("=" * 65)
    print(comp_df[["architecture", "train_accuracy", "val_accuracy", "test_accuracy", "test_f1", "test_roc_auc", "test_fpr", "test_fnr"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # Sliced Evaluations by Noise Condition & Machine Type on Test Set
    # --------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("COMPUTING SLICED EVALUATION BY NOISE & MACHINE TYPE")
    print("=" * 65)

    test_meta = metadata.iloc[test_idx].copy().reset_index(drop=True)
    test_y = y[test_idx]

    noise_records = []
    machine_records = []

    for item in model_catalog:
        mid = item["id"]
        mname = item["display_name"]
        probs, _ = model_predictions[mid]["Test"]

        # By Noise Condition (-6_dB, 0_dB, 6_dB)
        for cond in ["-6_dB", "0_dB", "6_dB"]:
            mask = (test_meta["condition"] == cond).values
            m = calc_metrics(test_y[mask], probs[mask], threshold=0.5)
            m["condition"] = cond
            m["model_id"] = mid
            m["model_name"] = mname
            noise_records.append(m)

        # By Machine Type (fan, pump, slider, valve)
        for mach in ["fan", "pump", "slider", "valve"]:
            mask = (test_meta["machine"] == mach).values
            m = calc_metrics(test_y[mask], probs[mask], threshold=0.5)
            m["machine"] = mach
            m["model_id"] = mid
            m["model_name"] = mname
            machine_records.append(m)

    noise_df = pd.DataFrame(noise_records)
    noise_csv = os.path.join(METRICS_DIR, "noise_condition_breakdown.csv")
    noise_df.to_csv(noise_csv, index=False)

    machine_df = pd.DataFrame(machine_records)
    machine_csv = os.path.join(METRICS_DIR, "machine_type_breakdown.csv")
    machine_df.to_csv(machine_csv, index=False)

    print("\n--- Test Accuracy & ROC-AUC by Noise Condition ---")
    print(noise_df[["condition", "model_id", "samples", "accuracy", "f1", "roc_auc", "false_alarm_rate_fpr", "missed_fault_rate_fnr"]].to_string(index=False))

    print("\n--- Test Accuracy & ROC-AUC by Machine Type ---")
    print(machine_df[["machine", "model_id", "samples", "accuracy", "f1", "roc_auc", "false_alarm_rate_fpr", "missed_fault_rate_fnr"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # Generate Figures
    # --------------------------------------------------------------------------
    generate_evaluation_figures(comp_df, noise_df, machine_df, model_predictions, test_y)

    print("\n" + "=" * 65)
    print("EVALUATION ENGINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)


def generate_evaluation_figures(comp_df, noise_df, machine_df, model_predictions, test_y):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Four-Panel Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=250)
    models_to_cm = [
        (axes[0, 0], "baseline_cnn", "Baseline CNN"),
        (axes[0, 1], "improved_cnn", "Improved Acoustic CNN"),
        (axes[1, 0], "baseline_cnn_gru", "Baseline CNN-GRU"),
        (axes[1, 1], "improved_cnn_gru", "Improved Acoustic CNN-GRU")
    ]

    for ax, mid, title in models_to_cm:
        probs, _ = model_predictions[mid]["Test"]
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(test_y, preds, labels=[0, 1])

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{title} (Test Acc: {accuracy_score(test_y, preds)*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal (0)", "Abnormal (1)"])
        ax.set_yticklabels(["Normal (0)", "Abnormal (1)"])
        ax.grid(False)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / np.sum(cm) * 100
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "experiment_confusion_matrices.png")
    plt.savefig(cm_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices figure to: {cm_path}")

    # 2. Headline Accuracy & F1 Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)
    x = np.arange(len(comp_df))
    width = 0.25

    ax.bar(x - width, comp_df["train_accuracy"] * 100, width, label="Train Accuracy", color="#6baed6", alpha=0.9)
    ax.bar(x, comp_df["val_accuracy"] * 100, width, label="Validation Accuracy", color="#3182bd", alpha=0.9)
    ax.bar(x + width, comp_df["test_accuracy"] * 100, width, label="Test Accuracy", color="#08519c", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" ", "\n") for r in comp_df["architecture"]], fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim([40, 100])
    ax.set_title("Training, Validation, and Test Accuracy Across Experiments (Leak-Free Split)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    comp_fig_path = os.path.join(FIGURES_DIR, "accuracy_comparison_bars.png")
    plt.savefig(comp_fig_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy comparison bars to: {comp_fig_path}")

    # 3. Test ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6), dpi=250)
    colors = {"baseline_cnn": "#1f77b4", "baseline_cnn_gru": "#ff7f0e", "improved_cnn": "#2ca02c", "improved_cnn_gru": "#d62728"}

    for item in model_catalog:
        mid = item["id"]
        probs, _ = model_predictions[mid]["Test"]
        fpr, tpr, _ = roc_curve(test_y, probs)
        auc = roc_auc_score(test_y, probs)
        ax.plot(fpr, tpr, color=colors[mid], lw=2.2, label=f"{item['display_name']} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (0.50)")
    ax.set_title("Test Set ROC Curves Across Experiments (Leak-Free Split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "experiment_roc_curves.png")
    plt.savefig(roc_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves figure to: {roc_path}")


if __name__ == "__main__":
    run_full_evaluation()
