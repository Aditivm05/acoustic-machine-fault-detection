# Final Operating Threshold Evaluation

**Project:** Acoustic Machine-Fault Detection Using CNN and CNN-GRU  
**Dataset:** MIMII  
**Selected model:** Improved Acoustic CNN-GRU  
**Selected operating threshold:** 0.26

## 1. Objective and threshold selection

The objective was to balance fault detection and false alarms. The operating threshold was selected using the validation set only by maximizing balanced accuracy. The selected threshold was **0.26**, with validation balanced accuracy of **89.25%**.

## 2. Validation results at threshold 0.26

| Metric | Validation result |
|---|---:|
| Balanced accuracy | 89.25% |
| Sensitivity | 91.62% |
| Specificity | 86.89% |
| Accuracy | 89.23% |
| Precision | 87.23% |
| F1-score | 89.37% |
| False-alarm rate | 13.11% |
| Missed-fault rate | 8.38% |
| ROC-AUC | 95.56% |

## 3. Held-out test results

The held-out test set was evaluated at the fixed threshold of 0.26.

| Metric | Test result |
|---|---:|
| Accuracy | 86.26% |
| F1-score | 86.84% |
| ROC-AUC | 94.66% |
| Sensitivity (fault detection) | 91.67% |
| Specificity (normal detection) | 80.98% |
| False-alarm rate | 19.02% |
| Missed-fault rate | 8.33% |

## 4. Test confusion matrix

Rows represent actual classes and columns represent predicted classes. Class 0 is normal and class 1 is fault.

| Actual class | Predicted normal | Predicted fault |
|---|---:|---:|
| Normal | 149 (TN) | 35 (FP) |
| Fault | 15 (FN) | 165 (TP) |

The model correctly detected 165 of 180 actual fault samples and missed 15. It correctly classified 149 of 184 normal samples, while 35 normal samples were flagged as faults.

## 5. Comparison with threshold 0.50

| Metric | Threshold 0.50 | Threshold 0.26 |
|---|---:|---:|
| Accuracy | 86.81% | 86.26% |
| F1-score | 85.88% | 86.84% |
| ROC-AUC | 94.66% | 94.66% |
| Sensitivity | 81.11% | 91.67% |
| Missed-fault rate | 18.89% | 8.33% |

Compared with threshold 0.50, threshold 0.26 increases sensitivity by 10.56 percentage points and reduces the missed-fault rate by 10.56 percentage points. F1-score increases by approximately 0.96 percentage points, while accuracy decreases by approximately 0.55 percentage points. ROC-AUC is unchanged because it measures ranking performance across thresholds, not performance at one fixed threshold.

## 6. Conclusion

Threshold **0.26** is retained because it was selected using validation balanced accuracy and aligns with the goal of reducing missed faults. On the held-out test set, the model achieved **91.67% sensitivity** and an **8.33% missed-fault rate**, with a **19.02% false-alarm rate**. The false-alarm rate should be considered when assessing suitability for deployment.

The threshold was selected on validation data and then evaluated on the held-out test set. Keep the threshold fixed and do not retune it using test-set results.
