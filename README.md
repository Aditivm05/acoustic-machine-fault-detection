# Acoustic Machine-Fault Detection Using CNN and CNN-GRU

An end-to-end deep-learning prototype that screens industrial machine audio for potentially abnormal operation. It converts WAV recordings into Log-Mel spectrograms, runs trained CNN-based models, and presents a screening result in a Streamlit web app.

**Live demo:** [Open the Streamlit application](https://acoustic-machine-fault-detection.streamlit.app/)  
**Repository:** [Aditivm05/acoustic-machine-fault-detection](https://github.com/Aditivm05/acoustic-machine-fault-detection)

> **Research prototype:** This application is intended for academic demonstration and screening. A “Normal Operation” result does not guarantee that a machine is fault-free, and results should not replace professional inspection or validated industrial safety systems.

---

## Project objectives

- Explore acoustic anomaly screening for industrial machine recordings.
- Convert audio into Log-Mel spectrogram features suitable for deep learning.
- Compare baseline CNN and CNN-GRU models with improved variants.
- Provide an interactive interface to upload WAV audio, run inference, and inspect the spectrogram.
- Examine the trade-off between missed faults and false alarms when selecting a decision threshold.

## Dataset and audio features

This project uses the **MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset**.

- **Machine types:** fan, pump, slider, and valve.
- **Noise conditions explored:** -6 dB, 0 dB, and +6 dB SNR.
- **Feature representation:** 128-band Log-Mel spectrograms.
- **Audio preprocessing:** resampling to 16 kHz and processing into fixed-duration segments.

The dataset is not bundled with this repository. Dataset availability, terms, and download instructions should be checked at the original MIMII dataset source before use. Training and evaluation scripts may require paths to be configured for your local dataset.

## Model architectures

### CNN

The convolutional neural network learns local frequency and time patterns in the Log-Mel spectrogram representation.

### CNN-GRU

The hybrid model combines convolutional feature extraction with a Gated Recurrent Unit (GRU) to model temporal relationships across acoustic frames.

The repository includes baseline and improved model variants. The deployed application uses the improved CNN and improved CNN-GRU implementations.

## Evaluation results

The following results are from the held-out test evaluation recorded for this project. Results depend on the dataset split, preprocessing, checkpoint, and decision threshold.

| Model / operating point | Accuracy | Precision | Recall / Sensitivity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 64.56% | 76.29% | 41.11% | 53.43% | 0.7225 |
| Baseline CNN-GRU | 74.18% | 75.60% | 70.56% | 72.99% | 0.8310 |
| Improved Acoustic CNN | 82.97% | 81.38% | 85.00% | 83.15% | 0.9139 |
| Improved Acoustic CNN-GRU, threshold 0.50 | 86.81% | 91.25% | 81.11% | 85.88% | 0.9466 |
| Improved Acoustic CNN-GRU, threshold 0.26 | 86.26% | — | 91.67% | 86.84% | 0.9466 |

At threshold **0.26**, test specificity was **80.98%**, the false-alarm rate was **19.02%**, and the missed-fault rate was **8.33%**. The threshold was selected using validation data, not the test set. Compared with threshold 0.50, threshold 0.26 increases sensitivity but also increases false alarms; the two operating points should not be treated as if every metric improves.

The threshold-0.26 test confusion matrix is `[[149, 35], [15, 165]]` in `[[TN, FP], [FN, TP]]` order.

Detailed report: [Final Operating Threshold Evaluation](experiments/accuracy_improvement_v1/metrics/final_threshold_evaluation.md)

## Try the deployed application

Open the [Streamlit app](https://acoustic-machine-fault-detection.streamlit.app/) and follow these steps:

1. Choose a WAV audio file.
2. Click **Analyze Uploaded Audio**.
3. Check the source filename and use the audio player if needed.
4. Click **Run Prediction**.
5. Review the abnormality probability, decision threshold, screening label, inference timing, and normalized Log-Mel spectrogram.

The label is produced by comparing the model probability with the configured threshold. It should be interpreted as a model screening result rather than a definitive diagnosis.

## Run locally

### Requirements

- Python 3.10 or a compatible Python version for the installed dependencies
- Git
- The trained checkpoint files included in the repository
- Optional: the MIMII dataset if you want to use dataset-backed sample browsing or reproduce training/evaluation

### Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/Aditivm05/acoustic-machine-fault-detection.git
cd acoustic-machine-fault-detection
```

Create and activate a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

Launch the application from the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`.

If you are running training or evaluation scripts, review their dataset path settings and configure them for your own MIMII data location. The hosted application and training pipeline have different resource requirements.

## Repository layout

```text
.
├── app.py
├── requirements.txt
├── src/
│   └── audio_processing.py
├── experiments/
│   └── accuracy_improvement_v1/
│       ├── models_v1.py
│       ├── train_experiments.py
│       ├── evaluate_experiments.py
│       ├── baseline_cnn_best.pt
│       ├── baseline_cnn_gru_best.pt
│       ├── improved_cnn_best.pt
│       ├── improved_cnn_gru_best.pt
│       ├── figures/
│       ├── history/
│       ├── metrics/
│       └── splits/
└── outputs/
    └── configs/
        └── threshold_config.json
```

Some generated reports or optional dataset directories may not be present in every checkout. The application expects checkpoint and configuration paths to match the repository layout.

## Reproducibility and data leakage

The recorded methodology uses grouped splitting by source recording ID to reduce the risk of related recordings appearing in more than one split. The recorded split contains 2,400 clips: 1,674 training, 362 validation, and 364 test clips, with no source-recording overlap between splits.

When reproducing results, use the same split definitions and preprocessing settings. Do not select a threshold using test labels; use validation data for threshold selection and reserve the test set for final evaluation.

## Limitations and future work

- The results depend on the machines, recording conditions, and noise levels represented in the dataset.
- Background noise and differences between dataset recordings and real industrial environments may affect predictions.
- A normal label is not proof of machine health; false alarms and missed faults are possible.
- Further testing on independent, real-world recordings is needed before considering operational use.

Potential next steps include testing additional machine recordings, improving robustness to domain shifts, reporting per-machine performance, and evaluating lightweight deployment on edge devices.

## Acknowledgements

The project uses the MIMII industrial machine sound dataset for academic experimentation. Please consult the original dataset source and follow its applicable terms when obtaining or redistributing data.
