<<<<<<< HEAD
# Acoustic Machine-Fault Detection Using CNN and CNN-GRU

An end-to-end deep learning project designed to detect anomalies and machine faults across various industrial equipment types (**fan, pump, slider, valve**) operating under three noise conditions (**-6 dB, 0 dB, +6 dB**). 

This system extracts frequency-domain representations via **Log-Mel Spectrograms** and evaluates structural variations using a baseline **CNN** and a sequential **CNN-GRU** architecture.

---

## 🛠️ Dataset & Feature Representation

- **Dataset Source:** MIMII (Industrial Machine Anomaly Detection Dataset).
- **Machine Types Analyzed:** Fan, Pump, Slider, Valve.
- **Signal-to-Noise Ratio (SNR) Conditions:** -6 dB (high background noise), 0 dB (moderate), +6 dB (low background noise).
- **Feature Processing:** Audio signals resampled to 16 kHz, windowed, and computed into **128-band Log-Mel Spectrograms** (5-second segments, 1024 FFT window size, 512 hop length).

---

## 🧪 Rigorous Leak-Free Methodology

To prevent data leakage (a common issue in acoustic detection where overlapping window frames or recordings from the same machine session end up in both training and test sets):
- **Grouped Splitting:** The data is split strictly by the **source recording ID** rather than random window cropping.
- **Zero-Overlap Guarantee:** 
  - **Total Samples:** 2,400
  - **Train Split:** 1,674 clips (683 distinct source recordings)
  - **Validation Split:** 362 clips (147 distinct source recordings)
  - **Test Split:** 364 clips (147 distinct source recordings)
  - **Inter-Split Overlap:** strictly **0** overlapping recordings.

---

## 📊 Model Architectures & Performance

Two principal deep learning configurations were built and optimized:
1. **Acoustic CNN:** Three convolutional blocks with Batch Normalization, ReLU activation, Max Pooling, and Global Average Pooling to reduce spatial dimensions.
2. **Acoustic CNN-GRU:** Integrates recurrent **Gated Recurrent Unit (GRU)** layers on top of the convolutional feature maps to model temporal dependencies across frames.

### Final Evaluation Comparison

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN (Leak-Free)** | 64.56% | 76.29% | 41.11% | 53.43% | 0.7225 |
| **Baseline CNN-GRU (Leak-Free)** | 74.18% | 75.60% | 70.56% | 72.99% | 0.8310 |
| **Improved Acoustic CNN** | 82.97% | 81.38% | 85.00% | 83.15% | 0.9139 |
| **Improved Acoustic CNN-GRU** | **86.81%** | **91.25%** | **81.11%** | **85.88%** | **0.9466** |

*The final selected architecture is the **Improved Acoustic CNN-GRU** owing to its robust temporal mapping capabilities.*

---

## 📁 Repository Structure

```text
.gitignore
README.md
experiments/
└── accuracy_improvement_v1/
    ├── checkpoints/           # Trained model state dict weights
    │   ├── baseline_cnn_best.pt
    │   ├── baseline_cnn_gru_best.pt
    │   ├── improved_cnn_best.pt
    │   └── improved_cnn_gru_best.pt
    ├── figures/               # Learning curves, ROC curves, confusion matrices
    ├── history/               # CSV training logs per epoch
    ├── metrics/               # Model comparison summaries & CSV reports
    ├── splits/                # Leak-free index splits & split_verification_report.json
    ├── models_v1.py           # PyTorch architecture class definitions
    ├── train_experiments.py   # Training loop orchestration script
    └── evaluate_experiments.py# Evaluation & performance visualization
```

---

## 🚀 Local Setup & Configuration

To replicate or run evaluations locally:
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install torch librosa soundfile scikit-learn matplotlib pandas tqdm
   ```
3. Configure dataset directory:  
   To avoid absolute paths, configure your root paths inside `train_experiments.py` or `evaluate_experiments.py` to target your local machine directory.
=======
# acoustic-machine-fault-detection
Acoustic machine-fault detection using CNN and CNN-GRU models on the MIMII dataset.
>>>>>>> 413bec2fa6ac681cad4dc545979f593cab770e02
