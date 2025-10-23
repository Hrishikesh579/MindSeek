# in integrated_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model1_audio.predict_audio_emotion import predict_audio_emotion
from model2_text.predict_text import predict_text as text_predict

def predict_audio(file_path):
    return predict_audio_emotion(file_path)

def predict_text(text):
    return text_predict(text)
