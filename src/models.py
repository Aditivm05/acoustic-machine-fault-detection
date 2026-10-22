"""
Neural Network Architectures for Acoustic Machine-Fault Detection
- FanCNN: 2D Convolutional Neural Network baseline
- FanCNNGRU: Hybrid CNN feature extractor with Bidirectional/Unidirectional GRU temporal modeling
"""

import torch
import torch.nn as nn


class FanCNN(nn.Module):
    """
    Standard 2D CNN baseline for acoustic fault detection.
    Processes Log-Mel spectrogram inputs of shape [batch_size, 1, 128, 157].
    Outputs raw scalar logits (use BCEWithLogitsLoss during training, sigmoid for inference).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.conv2(x)
        features = self.conv3(x)
        pooled = self.pool(features).flatten(1)
        logits = self.fc(self.dropout(pooled)).squeeze(1)
        if return_features:
            return logits, features
        return logits


class FanCNNGRU(nn.Module):
    """
    Hybrid CNN-GRU network for acoustic fault detection.
    Passes spectrograms through convolutional layers to extract frequency-time feature maps,
    averages over the frequency dimension to preserve temporal order, and passes the temporal
    sequence through a GRU to capture temporal dynamics of machine degradation/anomalies.
    Outputs raw scalar logits.
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.conv1 = FanCNN().conv1
        self.conv2 = FanCNN().conv2
        self.conv3 = FanCNN().conv3
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, 1, 128, 157]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [batch, 64, freq_dim, time_dim]
        # Pool across frequency dimension to form temporal feature vectors
        x = x.mean(dim=2)          # [batch, 64, time_dim]
        x = x.transpose(1, 2)      # [batch, time_dim, 64]
        sequence, _ = self.gru(x)  # [batch, time_dim, hidden_size]
        final_state = sequence[:, -1, :]  # Take final time step state
        return self.fc(self.dropout(final_state)).squeeze(1)


def load_model(model_name="CNN-GRU", checkpoint_path="cnn_gru_improved_best.pt", device="cpu"):
    """
    Helper function to instantiate and load weights safely onto the target device.
    """
    device = torch.device(device)
    if model_name.upper() in ["CNN", "FANCNN"]:
        model = FanCNN()
    elif model_name.upper() in ["CNN-GRU", "CNN_GRU", "FANCNNGRU"]:
        model = FanCNNGRU()
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
