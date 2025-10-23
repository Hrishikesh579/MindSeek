import torch
import torchaudio
from .train_model1_audio import AudioEmotionModel, EMOTIONS

MODEL_PATH = "model1_audio/model1_trained.pth"

def predict_audio_emotion(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioEmotionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.flatten()
    waveform = waveform[:160000] if len(waveform) > 160000 else torch.nn.functional.pad(waveform, (0, 160000 - len(waveform)))
    waveform = waveform.to(device).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(waveform)
        pred = torch.argmax(outputs, dim=1).item()
    return EMOTIONS[pred]
