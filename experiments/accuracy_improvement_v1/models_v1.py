"""
Neural Network Architectures for Accuracy Improvement Experiments (v1)
Includes:
1. BaselineFanCNN: Exact original 3-layer CNN
2. BaselineFanCNNGRU: Exact original CNN-GRU
3. ImprovedAcousticCNN: 4-layer CNN with Concat Pooling and regularized classifier head
4. ImprovedAcousticCNNGRU: 3-layer CNN with Frequency Projection, Bidirectional GRU, and Temporal Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# 1. Baseline FanCNN (Original reference)
# ------------------------------------------------------------------------------
class BaselineFanCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        pooled = self.pool(x).flatten(1)
        return self.fc(self.dropout(pooled)).squeeze(1)


# ------------------------------------------------------------------------------
# 2. Baseline FanCNNGRU (Original reference)
# ------------------------------------------------------------------------------
class BaselineFanCNNGRU(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.conv1 = BaselineFanCNN().conv1
        self.conv2 = BaselineFanCNN().conv2
        self.conv3 = BaselineFanCNN().conv3
        self.gru = nn.GRU(
            input_size=64, hidden_size=hidden_size, batch_first=True
        )
        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)          # Average across frequency: (batch, channels, time)
        x = x.transpose(1, 2)      # (batch, time, channels)
        sequence, _ = self.gru(x)
        final_state = sequence[:, -1, :]
        return self.fc(self.dropout(final_state)).squeeze(1)


# ------------------------------------------------------------------------------
# 3. Improved Acoustic CNN (4-Stage with Concat Pooling & Regularized Head)
# ------------------------------------------------------------------------------
class ImprovedAcousticCNN(nn.Module):
    """
    Enhanced CNN architecture designed for acoustic predictive maintenance:
    - 4 convolutional stages (32 -> 64 -> 128 -> 128) with LeakyReLU
    - Dual Concat-Pooling (Adaptive Avg + Max) capturing both background noise & acoustic peaks
    - Regularized classification head with Dropout (0.35 and 0.25) to prevent overfitting
    """
    def __init__(self, dropout_rate=0.35):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [32, 64, 78]
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [64, 32, 39]
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [128, 16, 19]
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [128, 8, 9]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Concat average and max pooling to retain peak energy anomalies
        avg_p = self.avg_pool(x).flatten(1)
        max_p = self.max_pool(x).flatten(1)
        pooled = torch.cat([avg_p, max_p], dim=1)
        
        return self.classifier(pooled).squeeze(1)


# ------------------------------------------------------------------------------
# 4. Improved Acoustic CNN-GRU (BiGRU + Temporal Attention)
# ------------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Calculates attention weights across time steps so the model can focus
    on intermittent faults rather than relying only on the final time frame.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch, time_steps, feature_dim]
        scores = self.attention(x)  # [batch, time_steps, 1]
        weights = F.softmax(scores, dim=1)  # [batch, time_steps, 1]
        context = torch.sum(weights * x, dim=1)  # [batch, feature_dim]
        return context, weights


class ImprovedAcousticCNNGRU(nn.Module):
    """
    Enhanced CNN-GRU architecture:
    - 3-stage convolutional front-end (32 -> 64 -> 128)
    - Frequency compression with 1x1 convolution
    - Bidirectional GRU (BiGRU) to capture forward and backward temporal progression
    - Temporal Attention pooling across the full 5-second acoustic duration
    """
    def __init__(self, gru_hidden=64, dropout_rate=0.35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [32, 64, 78]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [64, 32, 39]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # [128, 16, 19]
        )

        # Adaptive frequency pooling to 8 bins, then 1x1 conv to compress
        self.freq_pool = nn.AdaptiveAvgPool2d((8, None))
        self.freq_conv = nn.Conv1d(128 * 8, 96, kernel_size=1)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        bi_dim = gru_hidden * 2  # 128 features (forward + backward)

        # Temporal attention over all time frames
        self.attention = TemporalAttention(bi_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(bi_dim, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(48, 1)
        )

    def forward(self, x, return_attention=False):
        # x: [batch, 1, 128, 157]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [batch, 128, freq_bins=16, time_frames=19]

        x = self.freq_pool(x)  # [batch, 128, 8, 19]
        batch, c, f, t = x.shape
        x = x.view(batch, c * f, t)  # [batch, 1024, 19]
        x = F.leaky_relu(self.freq_conv(x), negative_slope=0.1)  # [batch, 96, 19]
        x = x.transpose(1, 2)  # [batch, time_frames=19, channels=96]

        gru_out, _ = self.gru(x)  # [batch, 19, 128]
        context, attn_weights = self.attention(gru_out)  # [batch, 128]

        logits = self.classifier(context).squeeze(1)
        if return_attention:
            return logits, attn_weights
        return logits
