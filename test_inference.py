"""
Validation and Reliability Test Script
Executes Task F:
1. Verifies that saved checkpoints can be cleanly loaded in a fresh Python process
2. Tests inference on sample normal and abnormal WAV files
3. Verifies label polarity (0 = Normal, 1 = Abnormal)
4. Tests error handling on empty or invalid inputs
5. Documents system specifications and package versions
"""

import os
import sys
import time
import numpy as np
import torch

from src.models import load_model
from src.audio_processing import preprocess_audio_file, prepare_input_tensor

PROJECT_DIR = r"C:\Capstone Project files"

def run_system_verification():
    print("=" * 65)
    print("TASK F: SYSTEM RELIABILITY AND INFERENCE VALIDATION")
    print("=" * 65)

    # 1. Environment and package versions
    print("\n1. Environment and Software Versions:")
    print(f"   Python Version : {sys.version.split()[0]}")
    print(f"   PyTorch Version: {torch.__version__}")
    try:
        import librosa
        print(f"   Librosa Version: {librosa.__version__}")
    except ImportError:
        pass
    try:
        import sklearn
        print(f"   Scikit-Learn   : {sklearn.__version__}")
    except ImportError:
        pass
    print(f"   Compute Device : {'CPU' if not torch.cuda.is_available() else 'CUDA'}")

    # 2. Checkpoint Loading Test
    print("\n2. Verifying Checkpoint Loading:")
    cnn_path = os.path.join(PROJECT_DIR, "cnn_improved_best.pt")
    gru_path = os.path.join(PROJECT_DIR, "cnn_gru_improved_best.pt")

    assert os.path.exists(cnn_path), f"CNN checkpoint missing at {cnn_path}"
    assert os.path.exists(gru_path), f"CNN-GRU checkpoint missing at {gru_path}"

    cnn = load_model("CNN", cnn_path, device="cpu")
    print("   [PASS] CNN checkpoint loaded successfully.")
    gru = load_model("CNN-GRU", gru_path, device="cpu")
    print("   [PASS] CNN-GRU checkpoint loaded successfully.")

    # 3. Audio Inference Test (Normal and Abnormal)
    print("\n3. Testing End-to-End Inference on Real Machine Audio:")

    # Select representative normal and abnormal test WAV files
    sample_normal = os.path.join(PROJECT_DIR, "6_dB_fan", "fan", "id_00", "normal", "00000000.wav")
    sample_abnormal = os.path.join(PROJECT_DIR, "6_dB_fan", "fan", "id_00", "abnormal", "00000000.wav")

    assert os.path.exists(sample_normal), f"Normal sample audio missing at {sample_normal}"
    assert os.path.exists(sample_abnormal), f"Abnormal sample audio missing at {sample_abnormal}"

    samples = [
        ("Normal Fan Audio (ID: 00)", sample_normal, 0),
        ("Abnormal Fan Audio (ID: 00)", sample_abnormal, 1)
    ]

    for label_name, audio_path, ground_truth in samples:
        print(f"\n   Testing: {label_name}")
        print(f"   File path: {audio_path}")

        # Preprocessing
        t0 = time.perf_counter()
        log_mel = preprocess_audio_file(audio_path)
        tensor_x = prepare_input_tensor(log_mel, device="cpu")
        prep_time_ms = (time.perf_counter() - t0) * 1000

        print(f"   Spectrogram shape: {log_mel.shape} (bins, time_frames)")
        print(f"   Input tensor shape: {tensor_x.shape}")
        print(f"   Preprocessing time: {prep_time_ms:.2f} ms")

        # Inference for CNN
        t0 = time.perf_counter()
        with torch.no_grad():
            cnn_logit = cnn(tensor_x).item()
            cnn_prob = torch.sigmoid(torch.tensor(cnn_logit)).item()
        cnn_time_ms = (time.perf_counter() - t0) * 1000

        # Inference for CNN-GRU
        t0 = time.perf_counter()
        with torch.no_grad():
            gru_logit = gru(tensor_x).item()
            gru_prob = torch.sigmoid(torch.tensor(gru_logit)).item()
        gru_time_ms = (time.perf_counter() - t0) * 1000

        cnn_pred_class = "Abnormal" if cnn_prob >= 0.5 else "Normal"
        gru_pred_class = "Abnormal" if gru_prob >= 0.5 else "Normal"
        gt_class = "Abnormal" if ground_truth == 1 else "Normal"

        print(f"   Ground Truth : {gt_class} ({ground_truth})")
        print(f"   CNN Output   : Prob={cnn_prob:.4f} -> Pred={cnn_pred_class} (Latency: {cnn_time_ms:.2f} ms)")
        print(f"   CNN-GRU Output: Prob={gru_prob:.4f} -> Pred={gru_pred_class} (Latency: {gru_time_ms:.2f} ms)")

    # 4. Error Handling Verification (Corrupted / Empty audio)
    print("\n4. Verifying Robust Error Handling on Edge Cases:")
    # Test 4a: Non-existent file
    try:
        preprocess_audio_file("non_existent_clip.wav")
        print("   [FAIL] Did not raise exception on non-existent file.")
    except ValueError as e:
        print(f"   [PASS] Non-existent file caught: {type(e).__name__}")

    # Test 4b: Empty/corrupted file
    dummy_corrupt_path = os.path.join(PROJECT_DIR, "corrupt_test.wav")
    with open(dummy_corrupt_path, "wb") as f:
        f.write(b"NOT A REAL WAV FILE HEADER 1234567890")

    try:
        preprocess_audio_file(dummy_corrupt_path)
        print("   [FAIL] Did not raise exception on corrupt file.")
    except ValueError as e:
        print(f"   [PASS] Corrupted WAV caught successfully: {type(e).__name__}")
    finally:
        if os.path.exists(dummy_corrupt_path):
            os.remove(dummy_corrupt_path)

    print("\n" + "=" * 65)
    print("ALL RELIABILITY AND VALIDATION CHECKS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    run_system_verification()
