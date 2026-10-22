"""
Streamlit Web Application: Acoustic Machine-Fault Detection Demonstration
Screening Prototype for Predictive Maintenance Using CNN and CNN-GRU Models
"""

import os
import json
import time
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import streamlit as st

from src.models import load_model, FanCNN, FanCNNGRU
from experiments.accuracy_improvement_v1.models_v1 import ImprovedAcousticCNNGRU
from src.audio_processing import (
    preprocess_audio_file, prepare_input_tensor, SR, DURATION, N_MELS, N_FFT, HOP_LENGTH
)

TRAIN_MEAN = 0.6007688045501709
TRAIN_STD = 0.14802886545658112
TRAIN_SCALER_EPSILON = 1e-6

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Acoustic Machine-Fault Detection",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #92400e;
        font-size: 0.92rem;
    }
    .status-normal {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        color: #065f46;
    }
    .status-abnormal {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        color: #991b1b;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_DIR = r"C:\Capstone Project files"
CONFIG_PATH = os.path.join(PROJECT_DIR, "outputs", "configs", "threshold_config.json")

# ------------------------------------------------------------------------------
# Model Caching
# ------------------------------------------------------------------------------
@st.cache_resource
def get_model(model_name: str):
    """Load and cache the selected model."""
    if model_name == "CNN Baseline":
        ckpt = os.path.join(PROJECT_DIR, "cnn_improved_best.pt")
        return load_model("CNN", ckpt, device="cpu")

    ckpt = os.path.join(
        PROJECT_DIR,
        "experiments",
        "accuracy_improvement_v1",
        "checkpoints",
        "improved_cnn_gru_best.pt",
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Final CNN-GRU checkpoint not found: {ckpt}")

    model = ImprovedAcousticCNNGRU()
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
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

if model_key == "CNN-GRU":
    threshold_mode = st.sidebar.radio(
        "Operating Strategy",
        ["Default Threshold (0.50)", "Custom Slider"],
        index=0,
        help="The improved CNN-GRU currently uses a transparent default threshold. "
             "A model-specific optimized threshold has not yet been validated."
    )
    if threshold_mode == "Default Threshold (0.50)":
        active_threshold = 0.50
        st.sidebar.info(
            "Default threshold: **0.50**. "
            "The improved model's low-alarm threshold has not yet been validated."
        )
    else:
        active_threshold = st.sidebar.slider(
            "Custom Decision Threshold", 0.05, 0.95, 0.50, 0.01
        )
else:
    threshold_mode = st.sidebar.radio(
        "Operating Strategy",
        [
            "Industrial Low-Alarm (FPR ≤ 15%)",
            "Default Baseline (0.50)",
            "Max F1 Sensitivity",
            "Custom Slider"
        ],
        index=0,
        help="These saved threshold settings apply to the legacy CNN model."
    )
    if threshold_mode == "Industrial Low-Alarm (FPR ≤ 15%)":
        active_threshold = float(
            cfg.get("selected_constrained_threshold", 0.53)
        )
        fpr = cfg.get("constrained_metrics", {}).get(
            "false_alarm_rate_fpr"
        )
        if fpr is not None:
            st.sidebar.info(
                f"Threshold: **{active_threshold:.2f}** "
                f"(saved validation FPR: {fpr * 100:.1f}%)"
            )
        else:
            st.sidebar.info(f"Threshold: **{active_threshold:.2f}**")
    elif threshold_mode == "Default Baseline (0.50)":
        active_threshold = 0.50
        st.sidebar.info("Default probability threshold: **0.50**")
    elif threshold_mode == "Max F1 Sensitivity":
        active_threshold = float(cfg.get("best_f1_threshold", 0.43))
        st.sidebar.info(f"Saved CNN F1 threshold: **{active_threshold:.2f}**")
    else:
        active_threshold = st.sidebar.slider(
            "Custom Decision Threshold", 0.05, 0.95, 0.50, 0.01
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Preprocessing Specifications")
st.sidebar.markdown(f"""
- **Sample Rate:** {SR} Hz
- **Target Duration:** {DURATION} sec
- **Mel Filter Bins:** {N_MELS}
- **FFT Size / Hop:** {N_FFT} / {HOP_LENGTH}
- **Input Tensor:** `[1, 1, 128, 157]`
- **Inference Mode:** CPU-Only
""")

# ------------------------------------------------------------------------------
# Main Content: Input Audio Selection
# ------------------------------------------------------------------------------
tab_upload, tab_samples, tab_benchmark = st.tabs(["📁 Upload WAV Audio", "🎵 Dataset Benchmark Samples", "📊 Benchmark Performance"])

audio_bytes = None
audio_source_desc = ""

with tab_upload:
    st.markdown("##### Upload a 16 kHz WAV recording of industrial machinery:")
    uploaded_file = st.file_uploader(
        "Choose a .wav file",
        type=["wav"],
        help="Upload any WAV audio recording. Files longer than 5 seconds will be trimmed; shorter files will be zero-padded."
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_source_desc = f"Uploaded File: {uploaded_file.name}"

with tab_samples:
    st.markdown("##### Test with pre-verified audio clips from the MIMII benchmark dataset:")
    col_mach, col_cond, col_label = st.columns(3)
    
    with col_mach:
        sample_mach = st.selectbox("Machine Type", ["fan", "pump", "slider", "valve"], index=0)
    with col_cond:
        sample_cond = st.selectbox("Noise Level", ["6_dB (Low Noise)", "0_dB (Moderate)", "-6_dB (High Noise)"], index=0)
    with col_label:
        sample_label = st.selectbox("Ground Truth Condition", ["normal", "abnormal"], index=0)

    cond_clean = sample_cond.split()[0]
    sample_dir = os.path.join(PROJECT_DIR, f"{cond_clean}_{sample_mach}", sample_mach, "id_00", sample_label)
    
    available_files = []
    if os.path.isdir(sample_dir):
        available_files = [f for f in os.listdir(sample_dir) if f.endswith(".wav")][:20]

    if available_files:
        chosen_sample = st.selectbox("Select Sample Clip", available_files, index=0)
        sample_full_path = os.path.join(sample_dir, chosen_sample)
        if st.button("Load Selected Dataset Clip"):
            with open(sample_full_path, "rb") as f:
                audio_bytes = f.read()
            audio_source_desc = f"MIMII Sample: {sample_mach.upper()} ({cond_clean}, {sample_label.upper()}) - {chosen_sample}"
    else:
        st.warning(f"No audio clips found at: {sample_dir}")

with tab_benchmark:
    st.markdown("##### Measured Test Set Performance Comparison:")
    overall_csv = os.path.join(PROJECT_DIR, "outputs", "metrics", "overall_model_evaluation.csv")
    if os.path.exists(overall_csv):
        df_overall = pd.read_csv(overall_csv)
        st.dataframe(df_overall[df_overall["split"] == "Test"].style.format({
            "accuracy": "{:.2%}", "precision": "{:.2%}", "recall": "{:.2%}",
            "f1": "{:.2%}", "roc_auc": "{:.4f}", "false_alarm_rate_fpr": "{:.2%}",
            "missed_fault_rate_fnr": "{:.2%}", "mean_latency_ms": "{:.2f} ms"
        }))
    st.image(os.path.join(PROJECT_DIR, "outputs", "figures", "noise_condition_metrics.png"), caption="Performance Across Background Noise Conditions")

# ------------------------------------------------------------------------------
# Audio Processing & Prediction Execution
# ------------------------------------------------------------------------------
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

        # Keep log_mel unchanged for the existing [0, 1] visualization.
        if model_choice == "CNN Baseline":
            model_features = log_mel
        else:
            model_features = (
                log_mel - TRAIN_MEAN
            ) / (TRAIN_STD + TRAIN_SCALER_EPSILON)

        input_tensor = prepare_input_tensor(model_features, device="cpu")
        if tuple(input_tensor.shape) != (1, 1, 128, 157):
            raise ValueError(
                f"Unexpected model input shape: {tuple(input_tensor.shape)}"
            )

        prep_latency_ms = (time.perf_counter() - t_prep_start) * 1000

        # Run selected model
        model = get_model(model_choice)
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            logit = model(input_tensor).item()
            prob_abnormal = torch.sigmoid(torch.tensor(logit)).item()
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000

        # Classification decision based on active threshold
        is_abnormal = (prob_abnormal >= active_threshold)

        # ----------------------------------------------------------------------
        # Prediction Output Section
        # ----------------------------------------------------------------------
        st.markdown("### 🔍 Diagnostic Screening Result")
        
        col_res, col_metrics = st.columns([1.2, 1.8])

        with col_res:
            if is_abnormal:
                st.markdown(f"""
                <div class="status-abnormal">
                    <h2 style="margin:0; font-size:1.8rem;">🚨 ABNORMAL FAULT DETECTED</h2>
                    <p style="margin-top:8px; font-size:1.0rem;">
                        Acoustic anomaly detected. The sound signature deviates significantly from baseline healthy operational harmonics.
                    </p>
                    <div style="font-size:1.1rem; font-weight:600; margin-top:12px;">
                        Abnormal Probability: {prob_abnormal*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-normal">
                    <h2 style="margin:0; font-size:1.8rem;">✅ NORMAL OPERATION</h2>
                    <p style="margin-top:8px; font-size:1.0rem;">
                        Machine audio matches healthy acoustic vibration profile. No critical acoustic degradation detected.
                    </p>
                    <div style="font-size:1.1rem; font-weight:600; margin-top:12px;">
                        Abnormal Probability: {prob_abnormal*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_metrics:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Abnormal Prob</div>
                    <div class="metric-value">{prob_abnormal*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Decision Threshold</div>
                    <div class="metric-value">{active_threshold:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Inference Latency</div>
                    <div class="metric-value">{infer_latency_ms:.2f} ms</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Abnormal Probability Meter:**")
            st.progress(float(min(max(prob_abnormal, 0.0), 1.0)))
            st.caption(f"Decision Boundary: Values ≥ **{active_threshold:.2f}** flag as Abnormal. Preprocessing time: **{prep_latency_ms:.1f} ms**.")

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
