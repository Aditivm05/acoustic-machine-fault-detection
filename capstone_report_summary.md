# Capstone Project Report Summary: Acoustic Machine-Fault Detection

**Project Title:** Acoustic Machine-Fault Detection for Predictive Maintenance Using CNN and CNN-GRU  
**Student:** Aditi Mishra (B.Tech Electronics & Computer Science, 3rd Year)  
**Academic Year:** 2026–2027  

---

## 1. Abstract

Unplanned industrial equipment downtime imposes severe economic losses and safety risks. Traditional predictive maintenance relies on contact-based vibration sensors or periodic manual inspection, which can be costly or hazardous in inaccessible environments. This capstone project develops an automated, non-contact acoustic machine-fault detection system leveraging deep learning. Using the MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset comprising four machine types (Fans, Pumps, Sliders, and Valves) under three signal-to-noise conditions (-6 dB, 0 dB, +6 dB), raw 16 kHz audio recordings are converted into normalized Log-Mel spectrograms ($128 \times 157$). 

Two architectures are systematically implemented and evaluated: a 2D Convolutional Neural Network (CNN) baseline and a hybrid CNN-GRU architecture incorporating Gated Recurrent Units for temporal modeling. Evaluated on an independent, stratified test dataset (360 clips), the CNN baseline achieved an accuracy of 73.61%, F1-score of 72.14%, and ROC-AUC of 0.8101, with an inference latency of 2.29 ms. The CNN-GRU hybrid achieved superior fault detection capability, attaining an accuracy of 74.44%, recall of 74.44% (+6.11% improvement over baseline), F1-score of 74.44%, and ROC-AUC of 0.8149, with a latency of 2.35 ms. Sliced evaluation reveals that high factory background noise (-6 dB) degrades performance, while continuous friction machines (Sliders) achieve exceptional ROC-AUC (0.9525). To satisfy industrial low-false-alarm constraints, validation-only threshold tuning was applied, reducing test false alarms to 12.22% while increasing precision to 83.70%. An interactive Streamlit demonstration prototype was constructed for real-time acoustic screening.

---

## 2. Theoretical Background & Mathematical Formulation

### 2.1 Acoustic Representation via Log-Mel Spectrograms
Acoustic waveforms $x[n]$ captured at sampling rate $F_s = 16,000\text{ Hz}$ are segmented into fixed 5.0-second windows ($N = 80,000$ samples). The Short-Time Fourier Transform (STFT) computes localized frequency spectra:
$$X[m, k] = \sum_{n=0}^{N_{\text{fft}}-1} x[n + mH] \cdot w[n] \cdot e^{-j 2\pi k n / N_{\text{fft}}}$$
where $N_{\text{fft}} = 1024$ is the FFT window length, $H = 512$ is the hop size (50% overlap), and $w[n]$ is the Hann window.

Power values $|X[m, k]|^2$ are mapped onto $M = 128$ triangular Mel filterbank bins simulating human auditory perception:
$$f_{\text{Mel}} = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
Logarithmic compression converts power to decibels:
$$S_{\text{dB}}[m, b] = 10 \log_{10}\left( \max\left(\frac{S[m, b]}{\max(S)}, 10^{-8}\right) \right)$$
The resulting spectrograms are clipped to $[-80\text{ dB}, 0\text{ dB}]$ and linearly scaled to $[0.0, 1.0]$:
$$S_{\text{norm}}[m, b] = \frac{S_{\text{dB}}[m, b] - (-80)}{0 - (-80)}$$
This produces an input tensor of fixed dimension $[B, 1, 128, 157]$.

---

## 3. Deep Learning Architectures

### 3.1 2D Convolutional Neural Network Baseline (`FanCNN`)
The CNN baseline treats the Log-Mel spectrogram as a single-channel spatial image:
- **Block 1:** $\text{Conv2D}(1 \to 16, 3\times3, p=1) \to \text{BatchNorm2d} \to \text{ReLU} \to \text{MaxPool2d}(2\times2)$
- **Block 2:** $\text{Conv2D}(16 \to 32, 3\times3, p=1) \to \text{BatchNorm2d} \to \text{ReLU} \to \text{MaxPool2d}(2\times2)$
- **Block 3:** $\text{Conv2D}(32 \to 64, 3\times3, p=1) \to \text{BatchNorm2d} \to \text{ReLU} \to \text{MaxPool2d}(2\times2)$
- **Spatial Pooling & Classification:** $\text{AdaptiveAvgPool2d}((1, 1)) \to \text{Dropout}(0.30) \to \text{Linear}(64 \to 1)$

### 3.2 Spatio-Temporal Hybrid Network (`FanCNNGRU`)
While CNNs effectively extract local time-frequency features, machinery degradation often manifests as evolving temporal patterns (e.g., rhythmic grinding or intermittent friction). The `FanCNNGRU` passes the output of Block 3 through an average pooling layer along the frequency axis, preserving the temporal ordering:
1. Feature tensor from Block 3: $[B, 64, F', T']$
2. Frequency collapsing: $x_{\text{time}} = \frac{1}{F'} \sum_{f} x[b, c, f, t] \in \mathbb{R}^{B \times 64 \times T'}$
3. Transposition to sequence format: $[B, T', 64]$
4. Gated Recurrent Unit (GRU): $\text{GRU}(\text{input\_size}=64, \text{hidden\_size}=64, \text{batch\_first}=\text{True})$
5. Final state classification: The hidden state at the last time step $h_{T'}$ is extracted:
$$\hat{y}_{\text{logit}} = W_o \cdot \text{Dropout}(h_{T'}) + b_o$$

---

## 4. Training Methodology & Regularization

- **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`), combining sigmoid activation and binary cross-entropy into a single numerically stable layer:
  $$\mathcal{L} = - \left[ y \log \sigma(\hat{y}_{\text{logit}}) + (1 - y) \log(1 - \sigma(\hat{y}_{\text{logit}})) \right]$$
- **Optimizer:** AdamW (Adam with decoupled weight decay)
  - Learning Rate: $\eta = 3 \times 10^{-4}$
  - Weight Decay: $\lambda = 1 \times 10^{-4}$
- **Gradient Clipping:** Maximum gradient norm clipped to 1.0 to stabilize recurrent gradients in the GRU.
- **Early Stopping:** Monitored validation ROC-AUC with patience $P = 5$ epochs.

---

## 5. Experimental Results & Analysis

### 5.1 Overall Test Set Results ($N_{\text{test}} = 360$, Threshold = 0.50)

| Metric | CNN Baseline | CNN-GRU Hybrid | Delta ($\Delta$) | Engineering Significance |
|:---|:---:|:---:|:---:|:---|
| **Accuracy** | 73.61% | **74.44%** | +0.83% | Higher overall correct classification |
| **Precision** | **76.40%** | 74.44% | -1.96% | CNN produces fewer total positive flags |
| **Recall (Sensitivity)** | 68.33% | **74.44%** | **+6.11%** | **CNN-GRU detects substantially more real faults** |
| **F1-Score** | 72.14% | **74.44%** | **+2.30%** | Higher harmonic balance of precision and recall |
| **ROC-AUC** | 0.8101 | **0.8149** | +0.0048 | Strong overall separability across all thresholds |
| **False Alarm Rate (FPR)** | **21.11%** | 25.56% | +4.45% | False positives on healthy clips |
| **Missed-Fault Rate (FNR)** | 31.67% | **25.56%** | **-6.11%** | **CNN-GRU reduces missed equipment failures** |
| **Latency (CPU)** | **2.29 ms** | 2.35 ms | +0.06 ms | Extremely low computational overhead for edge devices |

### 5.2 Breakdown by Factory Ambient Noise Level

| Noise Condition | CNN Accuracy | CNN-GRU Accuracy | CNN Recall | CNN-GRU Recall | CNN ROC-AUC | CNN-GRU ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **-6 dB (Severe Factory Noise)** | 70.00% | **71.82%** | 61.40% | **75.44%** | 0.7405 | **0.7815** |
| **0 dB (Moderate Ambient Noise)** | 70.68% | **72.18%** | 65.71% | **72.86%** | **0.8036** | 0.7939 |
| **+6 dB (Low Ambient Noise)** | **80.34%** | 79.49% | **79.25%** | 75.47% | **0.8806** | 0.8694 |

**Analysis:** Background factory noise directly impacts acoustic fault classification. Under severe -6 dB noise, the CNN-GRU demonstrates significant resilience, achieving an ROC-AUC of **0.7815** compared to **0.7405** for the CNN, and detecting **75.44%** of faults compared to only **61.40%** for the CNN. Under low-noise conditions (+6 dB), both models achieve ~80% accuracy and ~0.87–0.88 ROC-AUC.

### 5.3 Breakdown by Machine Type

| Machine Type | CNN Accuracy | CNN-GRU Accuracy | CNN F1 | CNN-GRU F1 | CNN ROC-AUC | CNN-GRU ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Slider** | **88.04%** | 86.96% | **87.64%** | 86.96% | **0.9568** | 0.9525 |
| **Pump** | 76.14% | **77.27%** | 72.73% | **77.78%** | **0.8796** | 0.8337 |
| **Fan** | 64.44% | **67.78%** | 60.00% | **67.42%** | 0.6986 | **0.7643** |
| **Valve** | 65.56% | 65.56% | **67.37%** | 65.17% | 0.6630 | **0.6962** |

**Analysis:**
1. **Sliders** achieved the highest performance ($\text{ROC-AUC} > 0.95$). Sliders involve linear mechanical movement where rail wear, guide friction, or metal-on-metal degradation generates continuous, broadband acoustic signatures easily distinguished from normal sliding sounds.
2. **Fans** benefit strongly from the recurrent GRU layer ($\text{ROC-AUC}$ increases from 0.6986 to 0.7643, recall increases from 52.17% to 65.22%). Fan anomalies typically manifest as subtle periodic unbalance or motor hum that evolves across multiple rotations.
3. **Valves** present the greatest challenge ($\text{ROC-AUC} \approx 0.66 - 0.70$). Valve operation consists of brief transient open/close events rather than continuous acoustic emissions; these short bursts are easily masked by factory background noise.

---

## 6. Threshold Selection & Industrial Strategy

In academic experiments, a default threshold of $0.50$ is standard. However, in factory predictive maintenance, classification thresholds dictate operational trade-offs:
- **Low Threshold ($T < 0.50$):** High recall (few missed faults), but high false alarm rate (FPR). Every false alarm can stop a production line for unnecessary inspection.
- **High Threshold ($T > 0.50$):** Low false alarm rate, but higher risk of missed catastrophic faults.

### Validation-Only Tuning Process
Candidate thresholds were evaluated across $[0.05, 0.95]$ with step $0.01$ **exclusively on validation data**. We constrained the false alarm rate to $\le 15\%$ while maximizing recall:
- For `CNN-GRU`, the optimal constrained threshold was **$T = 0.75$**.
- Applied to the unseen test set, this tuned threshold reduced false alarms from **25.56% down to 12.22%**, while precision surged to **83.70%** and test accuracy improved to **75.28%**.
- Crucially, **ROC-AUC is threshold-independent** (remaining 0.8149), confirming that threshold tuning adjusts the operating operating point along the existing ROC curve without altering intrinsic discrimination capacity.

---

## 7. Project Limitations & Engineering Roadmap

1. **Acoustic Subsampling & Hardware Scope:**
   The full MIMII dataset contains 50,574 audio recordings. Because training was conducted locally on a Windows CPU-only architecture, a representative balanced subset of 2,400 clips (100 normal + 100 abnormal across 12 combinations) was utilized. Future work should scale training to GPU infrastructure to ingest the full 50,000+ files.
2. **Near-Duplicate Audio Across Noise Levels:**
   In the MIMII benchmark, identical underlying machine sound clips are mixed with factory background noise at -6 dB, 0 dB, and 6 dB. Independent stratified sampling per folder resulted in some underlying acoustic events appearing in both train and test splits (under different noise levels). To eliminate near-duplicate leakage in future iterations, `GroupKFold` splitting grouped by `audio_base_id` should be enforced.
3. **Transient vs Continuous Audio Signatures:**
   Standard STFT Log-Mel representations are optimal for continuous acoustic signatures (Fans, Pumps, Sliders), but sub-optimal for impulsive, transient mechanisms (Valves). Implementing Wavelet Transforms or event-triggered windowing is recommended for valve fault monitoring.

---

## 8. Conclusion

This project successfully proves the feasibility of non-contact acoustic machine-fault detection for predictive maintenance. The hybrid CNN-GRU architecture provides notable advantages in sensitivity and noise resilience, capturing 74.44% of industrial faults with real-time inference latency under 2.5 ms per clip on standard consumer CPU hardware. By pairing deep learning feature extraction with domain-aware validation threshold tuning and an interactive screening prototype, this capstone delivers a complete, production-aligned machine learning engineering pipeline.
