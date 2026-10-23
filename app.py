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

# ------------------------------------------------------------------
# Main application tabs
# ------------------------------------------------------------------

tab_upload, tab_samples, tab_benchmark = st.tabs([
    "📁 Upload WAV Audio",
    "🎵 Dataset Benchmark Samples",
    "📊 Benchmark Performance",
])

# Shared audio state persisted across Streamlit reruns
audio_bytes = st.session_state.get("audio_to_analyze")
audio_source_desc = st.session_state.get(
    "audio_source_to_analyze",
    "No audio selected",
)

# ------------------------------------------------------------------
# Tab 1: Upload audio and run prediction
# ------------------------------------------------------------------


with tab_upload:
    st.subheader("Upload machine audio")

    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
        key="audio_upload",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file.getvalue(), format="audio/wav")

        if st.button("Analyze Uploaded Audio", key="analyze_upload"):
            st.session_state["audio_to_analyze"] = uploaded_file.getvalue()
            st.session_state["audio_source_to_analyze"] = uploaded_file.name
            st.session_state["run_prediction"] = True

# ------------------------------------------------------------------
# Tab 2: Dataset benchmark samples
# ------------------------------------------------------------------

with tab_samples:
    st.subheader("MIMII Dataset Benchmark Samples")

    col_mach, col_cond, col_label = st.columns(3)

    with col_mach:
        sample_mach = st.selectbox(
            "Machine Type",
            ["fan", "pump", "slider", "valve"],
            key="sample_machine",
        )

    with col_cond:
        sample_cond = st.selectbox(
            "Noise Level",
            [
                "6_dB (Low Noise)",
                "0_dB (Moderate)",
                "-6_dB (High Noise)",
            ],
            key="sample_noise",
        )

    with col_label:
        sample_label = st.selectbox(
            "Ground Truth Condition",
            ["normal", "abnormal"],
            key="sample_label",
        )

    cond_clean = sample_cond.split()[0]

    sample_dir = (
        PROJECT_DIR
        / f"{cond_clean}_{sample_mach}"
        / sample_mach
        / "id_00"
        / sample_label
    )

    available_files = (
        sorted(sample_dir.glob("*.wav"))[:20]
        if sample_dir.is_dir()
        else []
    )

    if available_files:
        chosen_sample = st.selectbox(
            "Select Sample Clip",
            available_files,
            format_func=lambda path: path.name,
            key="chosen_sample",
        )

        if st.button("Load and Analyze Dataset Sample"):
            st.session_state["audio_to_analyze"] = (
                chosen_sample.read_bytes()
            )
            st.session_state["audio_source_to_analyze"] = (
                f"MIMII {sample_mach} | {cond_clean} | "
                f"{sample_label} | {chosen_sample.name}"
            )
            st.success("Sample loaded. Open the Upload WAV Audio tab to analyze it.")
    else:
        st.info(
            "Dataset audio files are not available in this deployment. "
            "You can still upload your own WAV file in the first tab."
        )

# ------------------------------------------------------------------
# Run prediction for the selected audio
# ------------------------------------------------------------------

audio_to_analyze = st.session_state.get("audio_to_analyze")
audio_source_desc = st.session_state.get(
    "audio_source_to_analyze", "No audio selected"
)

if audio_to_analyze is not None:
    st.markdown("---")
    st.subheader("🎧 Audio Playback & Analysis")
    st.write(f"**Source:** {audio_source_desc}")
    st.audio(audio_to_analyze, format="audio/wav")

    if st.button("Run Prediction", key="run_prediction"):
        try:
            # Preprocess audio
            prep_start = time.perf_counter()

            audio_buffer = io.BytesIO(audio_to_analyze)
            log_mel = preprocess_audio_file(audio_buffer)

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

            model = get_model(model_choice)
            model.eval()

            infer_start = time.perf_counter()

            with torch.no_grad():
                logits = model(input_tensor)
                probability = torch.sigmoid(logits).item()

            infer_latency_ms = (
                time.perf_counter() - infer_start
            ) * 1000

            is_abnormal = probability >= active_threshold

            st.markdown("### 🔍 Diagnostic Screening Result")

            result_col, metrics_col = st.columns([1.2, 1.8])

            with result_col:
                if is_abnormal:
                    st.error("🚨 ABNORMAL FAULT DETECTED")
                    st.write(
                        "An acoustic anomaly was detected. "
                        "Further inspection is recommended."
                    )
                else:
                    st.success("✅ NORMAL OPERATION")
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

            # Spectrogram visualization
            st.markdown("---")
            st.subheader("📈 Normalized Log-Mel Spectrogram")

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
            fig.colorbar(image, ax=ax, label="Normalized Energy")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as err:
            st.error(f"Audio processing or prediction failed: {err}")
            st.info(
                "Check the WAV file, dependencies, checkpoint paths, "
                "and model input shape."
            )

# ------------------------------------------------------------------
# Tab 3: Benchmark performance
# ------------------------------------------------------------------

with tab_benchmark:
    st.markdown("##### Measured Test Set Performance Comparison")

    overall_csv = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
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
                    use_container_width=True,
                )
            else:
                st.info("No test-split rows were found in the metrics file.")
        else:
            st.warning("The metrics file has no 'split' column.")
    else:
        st.info(
            "Benchmark metrics are unavailable in this deployment."
        )

    benchmark_figure = (
        PROJECT_DIR
        / "outputs"
        / "figures"
        / "noise_condition_metrics.png"
    )

    if benchmark_figure.is_file():
        st.image(
            str(benchmark_figure),
            caption="Performance Across Background Noise Conditions",
            use_container_width=True,
        )
    else:
        st.info("The noise-condition benchmark figure is unavailable.")



# ------------------------------------------------------------------
# Audio Processing and Prediction
# ------------------------------------------------------------------

if (
    audio_bytes is not None
    and st.session_state.get("run_prediction", False)
):
    st.markdown("---")
    st.subheader("🎧 Audio Playback & Analysis")
    st.write(f"**Source:** {audio_source_desc}")
    st.audio(audio_bytes, format="audio/wav")

    try:
        st.session_state["run_prediction"] = False
        # Preprocess audio
        prep_start = time.perf_counter()

        audio_buffer = io.BytesIO(audio_bytes)
        log_mel = preprocess_audio_file(audio_buffer)

        # Training-split normalization for improved models
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

        # Load model and predict
        model = get_model(model_choice)

        infer_start = time.perf_counter()

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()

        infer_latency_ms = (
            time.perf_counter() - infer_start
        ) * 1000

        is_abnormal = probability >= active_threshold

        # Display prediction
        st.markdown("### 🔍 Diagnostic Screening Result")

        result_col, metrics_col = st.columns([1.2, 1.8])

        with result_col:
            if is_abnormal:
                st.error("🚨 ABNORMAL FAULT DETECTED")
                st.write(
                    "An acoustic anomaly was detected. "
                    "Further inspection is recommended."
                )
            else:
                st.success("✅ NORMAL OPERATION")
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

        # Spectrogram visualization
        st.markdown("---")
        st.subheader("📈 Normalized Log-Mel Spectrogram")

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
        st.error(f"Audio processing or prediction failed: {err}")
        st.info(
            "Check the WAV file, dependencies, checkpoint paths, "
            "and model input shape."
        )

else:
    st.info(
        "👈 Upload an audio file or load a dataset sample to run "
        "the acoustic screening demonstration."
    )
