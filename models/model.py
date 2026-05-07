import torch
import torch.nn as nn
import torchvision.models as models

class ViolenceModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, dropout=0.4):
        super().__init__()

        # EfficientNet-B0: lighter and more accurate than ResNet18
        eff = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(eff.children())[:-2])  # remove avg pool + head
        self.pool = nn.AdaptiveAvgPool2d(1)

        cnn_out = 1280  # EfficientNet-B0 output channels

        self.proj = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM sees context from both directions
        self.lstm = nn.LSTM(
            512, hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feat = self.cnn(x)          # (B*T, 1280, h, w)
        feat = self.pool(feat).flatten(1)   # (B*T, 1280)
        feat = self.proj(feat)      # (B*T, 512)
        feat = feat.view(B, T, -1)  # (B, T, 512)

        out, _ = self.lstm(feat)    # (B, T, hidden*2)
        out = self.classifier(out[:, -1])   # use last timestep
        return out