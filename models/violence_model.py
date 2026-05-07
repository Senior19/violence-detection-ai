import torch
import torch.nn as nn
import torchvision.models as models


class ViolenceModel(nn.Module):

    def __init__(self):
        super(ViolenceModel, self).__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        b, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)

        features = []

        for i in range(t):
            frame = x[:, i]
            f = self.cnn(frame)
            features.append(f)

        features = torch.stack(features, dim=1)

        lstm_out, _ = self.lstm(features)

        out = lstm_out[:, -1, :]

        out = self.fc(out)

        return out