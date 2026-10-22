"""
Audio Preprocessing Pipeline for Acoustic Machine-Fault Detection
Preserves the exact parameters and normalization from training:
- Sampling rate: 16,000 Hz
- Target duration: 5.0 seconds (80,000 samples)
- Mel frequency bins: 128
- FFT window size: 1024
- Hop length: 512
- Decibel range: [-80 dB, 0 dB], normalized to [0, 1]
- Output tensor shape: [batch_size, 1, 128, 157]
"""

import soundfile as sf
import librosa
import numpy as np
import torch

SR = 16000
DURATION = 5.0
TARGET_SAMPLES = int(SR * DURATION)  # 80,000 samples
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
DB_FLOOR = -80.0
DB_CEILING = 0.0


def waveform_to_logmel(audio: np.ndarray) -> np.ndarray:
    """
    Converts 1D raw waveform array into a normalized Log-Mel spectrogram.
    Padding/truncation ensures fixed temporal dimension of 157 frames.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Ensure audio is 1D (mono)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Pad with zeros or truncate to exact target length
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_SAMPLES]

    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Convert power to decibels relative to peak
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Clip to specified dynamic range [-80, 0] dB
    log_mel = np.clip(log_mel, DB_FLOOR, DB_CEILING)

    # Normalize to [0.0, 1.0]
    log_mel = (log_mel - DB_FLOOR) / (DB_CEILING - DB_FLOOR)

    return log_mel.astype(np.float32)


def preprocess_audio_file(file_or_path) -> np.ndarray:
    """
    Loads an audio file from path or file-like buffer, resamples to 16kHz mono,
    and returns a Log-Mel spectrogram of shape [128, 157].
    Raises descriptive ValueError for corrupted or unsupported files.
    """
    try:
        # First attempt fast direct read with soundfile
        audio, sample_rate = sf.read(file_or_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if sample_rate != SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
    except Exception as sf_err:
        try:
            # Fallback to librosa if soundfile failed
            audio, sample_rate = librosa.load(file_or_path, sr=SR, mono=True, duration=DURATION)
        except Exception as e:
            raise ValueError(f"Could not read audio file. Verify it is a valid, uncorrupted WAV file. Details: {e}")

    if len(audio) == 0:
        raise ValueError("Audio file contains 0 audio samples (empty file).")

    return waveform_to_logmel(audio)


def prepare_input_tensor(log_mel: np.ndarray, device="cpu") -> torch.Tensor:
    """
    Transforms [128, 157] log-mel spectrogram into [1, 1, 128, 157] PyTorch tensor.
    """
    tensor = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(device)
