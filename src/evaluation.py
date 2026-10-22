"""
Evaluation Engine for Acoustic Machine-Fault Detection
Computes:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (threshold-independent metric computed on probabilities)
- Confusion Matrix (TN, FP, FN, TP)
- False Alarm Rate (FPR) = FP / (FP + TN)
- Missed-Fault Rate (FNR) = FN / (FN + TP)
- Per-clip inference latency (ms/clip)
"""

import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def compute_predictions(
    model: nn.Module,
    features: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32
) -> Tuple[np.ndarray, float]:
    """
    Runs model inference over features of shape [N, 128, 157].
    Returns:
        probabilities: np.ndarray of shape [N] (in range [0, 1])
        mean_latency_ms: float (latency per sample in milliseconds)
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    # Convert to tensor [N, 1, 128, 157]
    tensor_data = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs_list = []
    start_time = time.perf_counter()

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs_list.extend(probs)

    elapsed_s = time.perf_counter() - start_time
    mean_latency_ms = (elapsed_s * 1000.0) / max(len(features), 1)

    return np.asarray(probs_list, dtype=np.float32), mean_latency_ms


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    latency_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates full suite of classification and diagnostic metrics.
    Definitions:
        FPR (False Alarm Rate) = FP / (FP + TN)
        FNR (Missed-Fault Rate) = FN / (FN + TP)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    probabilities = np.asarray(probabilities).astype(float).ravel()
    predictions = (probabilities >= threshold).astype(int)

    # Compute confusion matrix with fixed labels [0, 1]
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate rates safely avoiding division by zero
    total_normal = tn + fp
    total_abnormal = tp + fn
    fpr = float(fp / total_normal) if total_normal > 0 else 0.0
    fnr = float(fn / total_abnormal) if total_abnormal > 0 else 0.0

    acc = float(accuracy_score(y_true, predictions))
    prec = float(precision_score(y_true, predictions, zero_division=0))
    rec = float(recall_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))

    # ROC-AUC is only defined if both classes are present in y_true
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))
    else:
        auc = float("nan")

    return {
        "samples": len(y_true),
        "normal_count": int(total_normal),
        "abnormal_count": int(total_abnormal),
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
        "true_positives": int(tp),
        "mean_latency_ms": latency_ms if latency_ms is not None else float("nan")
    }


def tune_threshold_on_validation(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    max_acceptable_fpr: float = 0.15
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates candidate thresholds across [0.05, 0.95] strictly on VALIDATION data.
    Finds:
    1. Threshold maximizing F1
    2. Threshold satisfying industrial constraint: FPR <= max_acceptable_fpr with highest recall
    Returns:
        selected_threshold: float
        tuning_df: pd.DataFrame
        summary: Dict
    """
    y_val = np.asarray(y_val).astype(int).ravel()
    thresholds = np.arange(0.05, 0.96, 0.01)
    records = []

    for t in thresholds:
        m = calculate_metrics(y_val, val_probs, threshold=float(t))
        records.append({
            "threshold": round(float(t), 2),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "false_alarm_rate_fpr": m["false_alarm_rate_fpr"],
            "missed_fault_rate_fnr": m["missed_fault_rate_fnr"]
        })

    df = pd.DataFrame(records)

    # 1. Best F1
    best_f1_idx = df["f1"].idxmax()
    best_f1_row = df.loc[best_f1_idx].to_dict()

    # 2. Best under FPR constraint
    constrained = df[df["false_alarm_rate_fpr"] <= max_acceptable_fpr]
    if len(constrained) > 0:
        # Sort by highest recall, then lowest FPR
        best_constrained_row = constrained.sort_values(
            by=["recall", "false_alarm_rate_fpr"], ascending=[False, True]
        ).iloc[0].to_dict()
    else:
        best_constrained_row = best_f1_row

    # Rationale: in predictive maintenance, false alarms stop plant operations,
    # so we prioritize keeping FPR <= 15% while maximizing fault recall.
    selected_t = float(best_constrained_row["threshold"])

    summary = {
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "best_f1_metrics": best_f1_row,
        "selected_constrained_threshold": selected_t,
        "constrained_metrics": best_constrained_row,
        "constraint_description": f"Maximum Recall subject to False Alarm Rate (FPR) <= {max_acceptable_fpr*100:.0f}%"
    }

    return selected_t, df, summary


def evaluate_subsets(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    group_col: str,
    threshold: float = 0.5,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Evaluates model performance across subsets (e.g. noise conditions or machine types)
    using a fixed split index (e.g. test_idx).
    """
    split_meta = metadata.iloc[indices].copy()
    split_X = X[indices]
    split_y = y[indices]

    probs, _ = compute_predictions(model, split_X, device=device)

    results = []
    groups = sorted(split_meta[group_col].unique())

    for grp in groups:
        mask = (split_meta[group_col] == grp).values
        grp_y = split_y[mask]
        grp_probs = probs[mask]

        m = calculate_metrics(grp_y, grp_probs, threshold=threshold)
        m[group_col] = grp
        results.append(m)

    # Reorder columns nicely
    df = pd.DataFrame(results)
    cols = [group_col, "samples", "normal_count", "abnormal_count", "accuracy",
            "precision", "recall", "f1", "roc_auc", "false_alarm_rate_fpr", "missed_fault_rate_fnr"]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols]
