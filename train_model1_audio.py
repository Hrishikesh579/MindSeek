import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# ---------------- Audio Emotion Classes ----------------
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# ---------------- Audio Emotion Model ----------------
class AudioEmotionModel(nn.Module):
    def __init__(self):
        super(AudioEmotionModel, self).__init__()
        self.fc1 = nn.Linear(160000, 128)   # match saved checkpoint
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 6)        # match saved checkpoint (6 classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------- Dataset Loader ----------------
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        waveform = waveform.flatten()
        waveform = waveform[:160000] if len(waveform) > 160000 else np.pad(waveform, (0, 160000 - len(waveform)))
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = self.labels[idx]
        return waveform, label
