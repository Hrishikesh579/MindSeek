import torch
import torch.nn as nn

class AudioEmotionModel(nn.Module):
    def __init__(self):
        super(AudioEmotionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 emotion classes
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
