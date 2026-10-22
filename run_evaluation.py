"""
Comprehensive Evaluation Pipeline for Acoustic Machine-Fault Detection
Executes Tasks B, C, and D:
- Task B: Overall Metrics across Train, Validation, and Test sets for CNN and CNN-GRU
- Task C: Sliced Evaluations by Noise Condition (-6dB, 0dB, +6dB) and Machine Type (Fan, Pump, Slider, Valve)
- Task D: Validation-only Threshold Tuning, Config Export, and Test Evaluation at Tuned Threshold
- Generates high-quality publication-ready graphs and confusion matrices
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.models import load_model, FanCNN, FanCNNGRU
from src.dataset import load_project_data
from src.evaluation import (
    compute_predictions, calculate_metrics, tune_threshold_on_validation, evaluate_subsets
)

PROJECT_DIR = r"C:\Capstone Project files"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
CONFIGS_DIR = os.path.join(OUTPUT_DIR, "configs")

for d in [METRICS_DIR, FIGURES_DIR, CONFIGS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cpu"
print(f"Running evaluation on {DEVICE}...")

# 1. Load Data
X, y, metadata, train_idx, val_idx, test_idx = load_project_data(PROJECT_DIR)
print(f"Data loaded successfully: {len(X)} clips total.")
print(f"Split sizes - Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

# 2. Load Improved Checkpoints
cnn_path = os.path.join(PROJECT_DIR, "cnn_improved_best.pt")
cnn_gru_path = os.path.join(PROJECT_DIR, "cnn_gru_improved_best.pt")

cnn = load_model("CNN", cnn_path, device=DEVICE)
cnn_gru = load_model("CNN-GRU", cnn_gru_path, device=DEVICE)
print("Loaded improved checkpoints for CNN and CNN-GRU.")

# ==============================================================================
# Task B: Comprehensive Overall Evaluation (Train, Val, Test at threshold = 0.5)
# ==============================================================================
print("\n" + "=" * 60)
print("TASK B: Computing Comprehensive Evaluation Metrics")
print("=" * 60)

splits = {
    "Train": train_idx,
    "Validation": val_idx,
    "Test": test_idx
}

models = {
    "CNN": cnn,
    "CNN-GRU": cnn_gru
}

overall_records = []
predictions_store = {}

for model_name, model in models.items():
    predictions_store[model_name] = {}
    for split_name, indices in splits.items():
        sub_X = X[indices]
        sub_y = y[indices]

        probs, latency_ms = compute_predictions(model, sub_X, device=DEVICE)
        predictions_store[model_name][split_name] = (probs, sub_y)

        metrics = calculate_metrics(sub_y, probs, threshold=0.5, latency_ms=latency_ms)
        metrics["model"] = model_name
        metrics["split"] = split_name
        overall_records.append(metrics)

overall_df = pd.DataFrame(overall_records)
# Reorder columns logically
key_cols = ["model", "split", "samples", "accuracy", "precision", "recall", "f1",
            "roc_auc", "false_alarm_rate_fpr", "missed_fault_rate_fnr", "mean_latency_ms",
            "true_negatives", "false_positives", "false_negatives", "true_positives", "threshold"]
overall_df = overall_df[key_cols]

overall_csv_path = os.path.join(METRICS_DIR, "overall_model_evaluation.csv")
overall_df.to_csv(overall_csv_path, index=False)
print(f"Saved Task B overall metrics to: {overall_csv_path}")

test_summary = overall_df[overall_df["split"] == "Test"].copy()
print("\n--- Test Set Performance Comparison (Threshold = 0.5) ---")
print(test_summary[["model", "accuracy", "precision", "recall", "f1", "roc_auc",
                    "false_alarm_rate_fpr", "missed_fault_rate_fnr", "mean_latency_ms"]].to_string(index=False))

# ==============================================================================
# Task D: Validation-only Threshold Tuning and Test Re-evaluation
# ==============================================================================
print("\n" + "=" * 60)
print("TASK D: Validation-Only Threshold Tuning")
print("=" * 60)

threshold_configs = {}
tuned_test_records = []

for model_name, model in models.items():
    val_probs, val_y = predictions_store[model_name]["Validation"]
    test_probs, test_y = predictions_store[model_name]["Test"]

    # Tune on validation data only (FPR constraint <= 15%)
    selected_t, tuning_df, summary = tune_threshold_on_validation(val_y, val_probs, max_acceptable_fpr=0.15)
    tuning_df.to_csv(os.path.join(METRICS_DIR, f"{model_name.lower().replace('-', '_')}_val_threshold_tuning.csv"), index=False)

    threshold_configs[model_name] = summary
    print(f"\n{model_name}:")
    print(f"   Validation Best F1 threshold: {summary['best_f1_threshold']} (Val F1: {summary['best_f1_metrics']['f1']:.4f})")
    print(f"   Selected (FPR <= 15%) threshold: {selected_t} (Val Recall: {summary['constrained_metrics']['recall']:.4f}, Val FPR: {summary['constrained_metrics']['false_alarm_rate_fpr']:.4f})")

    # Evaluate on test set with selected threshold
    test_metrics_tuned = calculate_metrics(test_y, test_probs, threshold=selected_t)
    test_metrics_tuned["model"] = model_name
    test_metrics_tuned["split"] = "Test (Tuned Threshold)"
    tuned_test_records.append(test_metrics_tuned)

config_path = os.path.join(CONFIGS_DIR, "threshold_config.json")
with open(config_path, "w") as f:
    json.dump(threshold_configs, f, indent=4)
print(f"\nSaved threshold configuration to: {config_path}")

tuned_test_df = pd.DataFrame(tuned_test_records)
tuned_test_df = tuned_test_df[key_cols]
tuned_test_csv = os.path.join(METRICS_DIR, "tuned_threshold_test_evaluation.csv")
tuned_test_df.to_csv(tuned_test_csv, index=False)

print("\n--- Test Performance with Validation-Tuned Threshold ---")
print(tuned_test_df[["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
                     "false_alarm_rate_fpr", "missed_fault_rate_fnr"]].to_string(index=False))

# ==============================================================================
# Task C: Sliced Evaluations by Noise Condition and Machine Type
# ==============================================================================
print("\n" + "=" * 60)
print("TASK C: Sliced Evaluation by Noise Condition & Machine Type")
print("=" * 60)

# Evaluate on the TEST set
noise_records = []
machine_records = []

for model_name, model in models.items():
    # 1. Noise conditions (-6_dB, 0_dB, 6_dB)
    df_noise = evaluate_subsets(
        model, X, y, metadata, test_idx, group_col="condition", threshold=0.5, device=DEVICE
    )
    df_noise["model"] = model_name
    noise_records.append(df_noise)

    # 2. Machine types (fan, pump, slider, valve)
    df_machine = evaluate_subsets(
        model, X, y, metadata, test_idx, group_col="machine", threshold=0.5, device=DEVICE
    )
    df_machine["model"] = model_name
    machine_records.append(df_machine)

noise_df = pd.concat(noise_records, ignore_index=True)
noise_csv_path = os.path.join(METRICS_DIR, "noise_condition_evaluation.csv")
noise_df.to_csv(noise_csv_path, index=False)
print(f"Saved Noise Condition evaluation to: {noise_csv_path}")

machine_df = pd.concat(machine_records, ignore_index=True)
machine_csv_path = os.path.join(METRICS_DIR, "machine_type_evaluation.csv")
machine_df.to_csv(machine_csv_path, index=False)
print(f"Saved Machine Type evaluation to: {machine_csv_path}")

print("\n--- Performance by Noise Condition (Test Set) ---")
print(noise_df[["condition", "model", "samples", "accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate_fpr", "missed_fault_rate_fnr"]].to_string(index=False))

print("\n--- Performance by Machine Type (Test Set) ---")
print(machine_df[["machine", "model", "samples", "accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate_fpr", "missed_fault_rate_fnr"]].to_string(index=False))

# ==============================================================================
# Task C & G: Graph Generation (High Resolution 300 DPI)
# ==============================================================================
print("\n" + "=" * 60)
print("Generating Publication-Quality Figures & Confusion Matrices")
print("=" * 60)

# Plot 1: Confusion Matrices (2x2 grid: CNN @0.5, CNN @tuned, CNN-GRU @0.5, CNN-GRU @tuned)
fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

plot_configs = [
    (axes[0, 0], "CNN (Threshold = 0.5)", test_summary[test_summary["model"] == "CNN"].iloc[0]),
    (axes[0, 1], f"CNN (Tuned T = {threshold_configs['CNN']['selected_constrained_threshold']})", tuned_test_df[tuned_test_df["model"] == "CNN"].iloc[0]),
    (axes[1, 0], "CNN-GRU (Threshold = 0.5)", test_summary[test_summary["model"] == "CNN-GRU"].iloc[0]),
    (axes[1, 1], f"CNN-GRU (Tuned T = {threshold_configs['CNN-GRU']['selected_constrained_threshold']})", tuned_test_df[tuned_test_df["model"] == "CNN-GRU"].iloc[0])
]

for ax, title, row in plot_configs:
    tn, fp, fn, tp = int(row["true_negatives"]), int(row["false_positives"]), int(row["false_negatives"]), int(row["true_positives"])
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Abnormal (1)"], fontsize=10)
    ax.set_yticklabels(["Normal (0)", "Abnormal (1)"], fontsize=10)
    ax.grid(False)

    # Annotate numbers and percentages
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / np.sum(cm) * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

plt.tight_layout()
cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrices to: {cm_fig_path}")

# Plot 2: Noise Condition Metrics Comparison (Accuracy, F1, ROC-AUC)
conditions = ["-6_dB", "0_dB", "6_dB"]
cond_labels = ["-6 dB (High Noise)", "0 dB (Moderate)", "+6 dB (Low Noise)"]
x = np.arange(len(conditions))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
metrics_to_plot = [
    ("accuracy", "Accuracy", (0.5, 0.95)),
    ("f1", "F1-Score", (0.5, 0.95)),
    ("roc_auc", "ROC-AUC", (0.5, 1.0))
]

for idx, (metric_key, metric_title, y_lim) in enumerate(metrics_to_plot):
    ax = axes[idx]
    cnn_vals = [noise_df[(noise_df["model"] == "CNN") & (noise_df["condition"] == c)][metric_key].values[0] for c in conditions]
    gru_vals = [noise_df[(noise_df["model"] == "CNN-GRU") & (noise_df["condition"] == c)][metric_key].values[0] for c in conditions]

    rects1 = ax.bar(x - width/2, cnn_vals, width, label="CNN Baseline", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width/2, gru_vals, width, label="CNN-GRU", color="#d95f02", alpha=0.9)

    ax.set_title(f"{metric_title} by Noise Condition", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylim(y_lim)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.legend(loc="lower right" if metric_key != "recall" else "upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
noise_fig_path = os.path.join(FIGURES_DIR, "noise_condition_metrics.png")
plt.savefig(noise_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved noise condition comparison to: {noise_fig_path}")

# Plot 3: Machine Type Metrics Comparison
machines = ["fan", "pump", "slider", "valve"]
mach_labels = ["Fan", "Pump", "Slider", "Valve"]
x_mach = np.arange(len(machines))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
for idx, (metric_key, metric_title, y_lim) in enumerate(metrics_to_plot):
    ax = axes[idx]
    cnn_vals = [machine_df[(machine_df["model"] == "CNN") & (machine_df["machine"] == m)][metric_key].values[0] for m in machines]
    gru_vals = [machine_df[(machine_df["model"] == "CNN-GRU") & (machine_df["machine"] == m)][metric_key].values[0] for m in machines]

    rects1 = ax.bar(x_mach - width/2, cnn_vals, width, label="CNN Baseline", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x_mach + width/2, gru_vals, width, label="CNN-GRU", color="#2ca02c", alpha=0.9)

    ax.set_title(f"{metric_title} by Machine Type", fontsize=12, fontweight="bold")
    ax.set_xticks(x_mach)
    ax.set_xticklabels(mach_labels, fontsize=10)
    ax.set_ylim(y_lim)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
machine_fig_path = os.path.join(FIGURES_DIR, "machine_type_metrics.png")
plt.savefig(machine_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved machine type comparison to: {machine_fig_path}")

# Plot 4: ROC Curves on Test Set
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
from sklearn.metrics import roc_curve

cnn_probs, test_y = predictions_store["CNN"]["Test"]
gru_probs, _ = predictions_store["CNN-GRU"]["Test"]

fpr_cnn, tpr_cnn, _ = roc_curve(test_y, cnn_probs)
fpr_gru, tpr_gru, _ = roc_curve(test_y, gru_probs)

cnn_auc = test_summary[test_summary["model"] == "CNN"]["roc_auc"].values[0]
gru_auc = test_summary[test_summary["model"] == "CNN-GRU"]["roc_auc"].values[0]

ax.plot(fpr_cnn, tpr_cnn, color="#2b5c8f", lw=2.5, label=f"CNN Baseline (AUC = {cnn_auc:.4f})")
ax.plot(fpr_gru, tpr_gru, color="#d95f02", lw=2.5, label=f"CNN-GRU (AUC = {gru_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Alarm Rate / False Positive Rate (FPR)", fontsize=11)
ax.set_ylabel("Detection Rate / True Positive Rate (Recall)", fontsize=11)
ax.set_title("Test Set Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=11, frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
roc_fig_path = os.path.join(FIGURES_DIR, "roc_curves.png")
plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved ROC curves to: {roc_fig_path}")

# Plot 5: Validation Threshold Tuning Curve
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
gru_tuning_df = pd.read_csv(os.path.join(METRICS_DIR, "cnn_gru_val_threshold_tuning.csv"))

ax.plot(gru_tuning_df["threshold"], gru_tuning_df["f1"], color="#2ca02c", lw=2.2, label="F1-Score")
ax.plot(gru_tuning_df["threshold"], gru_tuning_df["recall"], color="#1f77b4", lw=2.2, label="Recall (Detection Rate)")
ax.plot(gru_tuning_df["threshold"], gru_tuning_df["false_alarm_rate_fpr"], color="#d62728", lw=2.2, label="False Alarm Rate (FPR)")

# Highlight selected threshold
sel_t = threshold_configs["CNN-GRU"]["selected_constrained_threshold"]
ax.axvline(sel_t, color="black", linestyle=":", lw=2, label=f"Selected Threshold ({sel_t:.2f})")
ax.axhline(0.15, color="red", linestyle="--", alpha=0.5, label="Max Acceptable FPR (15%)")

ax.set_title("CNN-GRU Validation Data Threshold Tuning Trade-offs", fontsize=12, fontweight="bold")
ax.set_xlabel("Decision Threshold (Sigmoid Probability)", fontsize=11)
ax.set_ylabel("Metric Value", fontsize=11)
ax.set_xlim([0.05, 0.95])
ax.set_ylim([0.0, 1.02])
ax.legend(loc="center right", fontsize=9, frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
thresh_fig_path = os.path.join(FIGURES_DIR, "threshold_tuning_curves.png")
plt.savefig(thresh_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved threshold tuning curves to: {thresh_fig_path}")

print("\n" + "=" * 60)
print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
