"""
Streamlit Web Application: Acoustic Machine-Fault Detection Demonstration
Screening Prototype for Predictive Maintenance Using CNN and CNN-GRU Models
"""


import os
import sys
import json
import time
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st

from src.audio_processing import (
    preprocess_audio_file,
    prepare_input_tensor,
    SR, DURATION, N_MELS, N_FFT, HOP_LENGTH
)

PROJECT_DIR = Path(__file__).resolve().parent
EXP_DIR = PROJECT_DIR / "experiments" / "accuracy_improvement_v1"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from models_v1 import ImprovedAcousticCNNGRU, ImprovedAcousticCNN

CONFIG_PATH = PROJECT_DIR / "outputs" / "configs" / "threshold_config.json"


# ------------------------------------------------------------------------------
# Model Caching
# ------------------------------------------------------------------------------
@st.cache_resource
def get_model(model_name: str):
    """Load the selected improved model and its trained weights."""

    if model_name == "CNN Baseline":
        model = ImprovedAcousticCNN()
        checkpoint_path = EXP_DIR / "improved_cnn_best.pt"
    else:
        model = ImprovedAcousticCNNGRU()
        checkpoint_path = EXP_DIR / "improved_cnn_gru_best.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_data
def get_threshold_configs():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

threshold_configs = get_threshold_configs()

# ------------------------------------------------------------------------------
# Header & Safety Disclaimer
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🔊 Acoustic Machine-Fault Detection Prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Maintenance Screening using Log-Mel Spectrograms with CNN & CNN-GRU Architectures</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Academic Prototype & Early Screening Disclaimer:</strong><br>
    This demonstration is an experimental machine-fault screening prototype developed for research and educational purposes. 
    It is trained on the MIMII benchmark dataset (Fan, Pump, Slider, Valve) and designed to assist early predictive maintenance inspection. 
    It is <strong>not</strong> certified as an industrial fail-safe or automated emergency shutdown system.
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar: Model and Threshold Configuration
# ------------------------------------------------------------------------------
st.sidebar.header("🛠️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Architecture",
    ["CNN-GRU (Temporal Model)", "CNN Baseline"],
    index=0,
    help="CNN Baseline uses spatial 2D convolutions. CNN-GRU combines convolutions with a recurrent GRU to track temporal acoustic progression."
)

model_key = "CNN-GRU" if "GRU" in model_choice else "CNN"
cfg = threshold_configs.get(model_key, {})

st.sidebar.markdown("---")
st.sidebar.header("🎯 Decision Threshold")


threshold_mode = st.sidebar.radio(
    "Operating Strategy",
    [
        "Validation-selected threshold",
        "Default Baseline (0.50)",
        "Max F1 Sensitivity",
        "Custom Slider"
    ],
    index=0,
    help=(
        "The validation-selected threshold for the improved CNN-GRU "
        "is 0.26. The threshold changes the decision, not the model."
    )
)

if threshold_mode == "Validation-selected threshold":
    if model_key == "CNN-GRU":
        active_threshold = 0.26
    else:
        # No validation-selected CNN threshold was provided here.
        active_threshold = 0.50

    st.sidebar.info(
        f"Decision threshold: {active_threshold:.2f}"
    )

elif threshold_mode == "Default Baseline (0.50)":
    active_threshold = 0.50
    st.sidebar.info("Decision threshold: 0.50")

elif threshold_mode == "Max F1 Sensitivity":
    default_f1 = 0.50 if model_key == "CNN" else 0.26
    active_threshold = float(
        cfg.get("best_f1_threshold", default_f1)
    )
    st.sidebar.info(
        f"F1 operating threshold: {active_threshold:.2f}"
    )

else:
    active_threshold = st.sidebar.slider(
        "Custom Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.26 if model_key == "CNN-GRU" else 0.50,
        step=0.01
    )


# ------------------------------------------------------------------------------
# Main Content: Input Audio Selection
# ------------------------------------------------------------------------------
tab_upload, tab_samples, tab_benchmark = st.tabs(["📁 Upload WAV Audio", "🎵 Dataset Benchmark Samples", "📊 Benchmark Performance"])


if audio_bytes is not None:
    st.markdown("---")
    st.subheader("🎧 Audio Playback & Analysis")
    st.write(f"**Source:** `{audio_source_desc}`")
    st.audio(audio_bytes, format="audio/wav")

    try:
        # Preprocessing
        t_prep_start = time.perf_counter()
        buffer = io.BytesIO(audio_bytes)
        log_mel = preprocess_audio_file(buffer)

        train_mean = 0.6007688045501709
        train_std = 0.14802886545658112

        model_input = (
            log_mel - train_mean
        ) / (train_std + 1e-6)

        input_tensor = prepare_input_tensor(
            model_input.astype(np.float32),
            device="cpu",
        )

        prep_latency_ms = (
            time.perf_counter() - t_prep_start
        ) * 1000

        # Run selected model
        model = get_model(model_choice)

        t_infer_start = time.perf_counter()
        with torch.no_grad():
            logit = model(input_tensor).item()
            prob_abnormal = torch.sigmoid(
                torch.tensor(logit)
            ).item()

        infer_latency_ms = (
            time.perf_counter() - t_infer_start
        ) * 1000
        is_abnormal = prob_abnormal >= active_threshold

        st.markdown("### 🔍 Diagnostic Screening Result")

        col_res, col_metrics = st.columns([1.2, 1.8])

        with col_res:
            if is_abnormal:
                st.warning(
                    f"ABNORMAL FAULT DETECTED — "
                    f"Abnormal probability: {prob_abnormal:.1%}"
                )
            else:
                st.success(
                    f"NORMAL OPERATION — "
                    f"Abnormal probability: {prob_abnormal:.1%}"
                )
        with col_metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("Abnormal Probability", f"{prob_abnormal:.1%}")
            m2.metric("Decision Threshold", f"{active_threshold:.2f}")
            m3.metric("Inference Latency", f"{infer_latency_ms:.2f} ms")

        st.progress(
            float(min(max(prob_abnormal, 0.0), 1.0))
        )

        st.caption(
            f"Preprocessing time: {prep_latency_ms:.1f} ms"
        )

        # Display original log-Mel spectrogram
        st.markdown("---")
        st.subheader("📈 Normalized Log-Mel Spectrogram")

        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
        im = ax.imshow(
            log_mel,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        ax.set_title("128 Mel Bins × Time Frames")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")
        fig.colorbar(im, ax=ax, label="Normalized Energy")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as err:
        st.error(f"Audio processing or prediction failed: {err}")
        st.info(
            "Check the WAV file, dependencies, checkpoint paths, "
            "and model input shape."
        )

else:
    st.info(
        "Upload an audio file or load a dataset sample to run "
        "the acoustic screening demonstration."
    )

with tab_benchmark:
    st.markdown("##### Measured Test Set Performance Comparison:")

    overall_csv = (
        PROJECT_DIR / "outputs" / "metrics"
        / "overall_model_evaluation.csv"
    )

    if overall_csv.is_file():
        df_overall = pd.read_csv(overall_csv)

        if "split" in df_overall.columns:
            test_df = df_overall[
                df_overall["split"].astype(str).str.lower() == "test"
            ]

            if not test_df.empty:
                formatters = {
                    col: fmt
                    for col, fmt in {
                        "accuracy": "{:.2%}",
                        "precision": "{:.2%}",
                        "recall": "{:.2%}",
                        "f1": "{:.2%}",
                        "roc_auc": "{:.4f}",
                        "false_alarm_rate_fpr": "{:.2%}",
                        "missed_fault_rate_fnr": "{:.2%}",
                        "mean_latency_ms": "{:.2f} ms",
                    }.items()
                    if col in test_df.columns
                }

                st.dataframe(
                    test_df.style.format(formatters),
                    use_container_width=True
                )
            else:
                st.info("No test-split rows were found in the metrics file.")
        else:
            st.warning("The metrics file does not contain a 'split' column.")
    else:
        st.info(
            "Benchmark metrics are not included in this deployment yet. "
            "Model predictions can still be made in the Audio Upload tab."
        )

    benchmark_figure = (
        PROJECT_DIR / "outputs" / "figures"
        / "noise_condition_metrics.png"
    )

    if benchmark_figure.is_file():
        st.image(
            str(benchmark_figure),
            caption="Performance Across Background Noise Conditions",
            use_container_width=True
        )
    else:
        st.info(
            "The noise-condition benchmark figure is unavailable "
            "in the deployed repository."
        )


# ------------------------------------------------------------------------------
# Audio Processing & Prediction Execution
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Audio Processing & Prediction
# ------------------------------------------------------------------------------

if audio_bytes is not None:
    st.markdown("---")
    st.subheader("Audio Playback & Analysis")
    st.write(f"**Source:** {audio_source_desc}")
    st.audio(audio_bytes, format="audio/wav")

    try:
        # 1. Preprocess the uploaded audio
        prep_start = time.perf_counter()

        audio_buffer = io.BytesIO(audio_bytes)
        log_mel = preprocess_audio_file(audio_buffer)

        # Training-split normalization for the improved models
        train_mean = 0.6007688045501709
        train_std = 0.14802886545658112

        standardized_features = (
            log_mel - train_mean
        ) / (train_std + 1e-6)

        input_tensor = prepare_input_tensor(
            standardized_features.astype(np.float32),
            device="cpu",
        )

        prep_latency_ms = (
            time.perf_counter() - prep_start
        ) * 1000

        # 2. Load the selected model and predict
        model = get_model(model_choice)

        infer_start = time.perf_counter()

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()

        infer_latency_ms = (
            time.perf_counter() - infer_start
        ) * 1000

        is_abnormal = probability >= active_threshold

        # 3. Display prediction
        st.markdown("### Diagnostic Screening Result")

        result_col, metrics_col = st.columns([1.2, 1.8])

        with result_col:
            if is_abnormal:
                st.error("ABNORMAL FAULT DETECTED")
                st.write(
                    "An acoustic anomaly was detected. "
                    "Further inspection is recommended."
                )
            else:
                st.success("NORMAL OPERATION")
                st.write(
                    "No anomaly was flagged at the selected threshold."
                )

        with metrics_col:
            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Abnormal Probability",
                f"{probability:.1%}",
            )
            metric2.metric(
                "Decision Threshold",
                f"{active_threshold:.2f}",
            )
            metric3.metric(
                "Inference Latency",
                f"{infer_latency_ms:.2f} ms",
            )

        st.progress(
            float(min(max(probability, 0.0), 1.0))
        )

        st.caption(
            f"Preprocessing time: {prep_latency_ms:.1f} ms"
        )

        # 4. Display the original log-Mel spectrogram
        st.markdown("---")
        st.subheader("Normalized Log-Mel Spectrogram")

        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)

        image = ax.imshow(
            log_mel,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )

        ax.set_title("128 Mel Bins × Time Frames")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")

        fig.colorbar(
            image,
            ax=ax,
            label="Normalized Energy",
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as err:
        st.error(
            f"Audio processing or prediction failed: {err}"
        )
        st.info(
            "Check the audio file, preprocessing dependencies, "
            "checkpoint paths, and model input shape."
        )

else:
    st.info(
        "Upload an audio file or load a dataset sample to run "
        "the acoustic screening demonstration."
    )

        # ----------------------------------------------------------------------
        # Spectrogram Visualization
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Normalized Log-Mel Spectrogram Feature Representation")
        
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
        im = ax.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title("Input Feature: 128 Mel Bins × 157 Time Frames (Normalized [0, 1])", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time Frames (Hop: 512 samples, 16 kHz)", fontsize=9)
        ax.set_ylabel("Mel Frequency Bins", fontsize=9)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Energy", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as err:
        st.error(f"⚠️ Audio Processing Error: {str(err)}")
        st.info("Please ensure the file is an uncorrupted, standard 16 kHz WAV audio recording.")

else:
    st.info("👈 Please upload an audio file or select a dataset sample clip to run the acoustic screening demonstration.")
