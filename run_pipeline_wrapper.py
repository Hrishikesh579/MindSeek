from integration.run_pipeline import main as run_main
from model1_audio.predict_audio_emotion import predict_audio_emotion
from model2_text.predict_text import predict_text

# Wrapper for audio prediction
def run_pipeline_audio(audio_path):
    return predict_audio_emotion(audio_path)

# Wrapper for text prediction
def run_pipeline_text(text_input):
    return predict_text(text_input)
