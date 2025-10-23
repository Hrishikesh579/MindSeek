# integration/run_pipeline.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from integration.integrated_model import predict_text as _predict_text, predict_audio as _predict_audio

def run_pipeline(input_data, input_type="text"):
    if input_type == "text":
        return _predict_text(input_data)
    elif input_type == "audio":
        return _predict_audio(input_data)
    else:
        raise ValueError("input_type must be 'text' or 'audio'.")

if __name__ == "__main__":
    choice = input("1=Audio, 2=Text: ")
    if choice == "1":
        file_path = input("Path to .wav file: ")
        print("Emotion:", run_pipeline(file_path, input_type="audio"))
    elif choice == "2":
        text = input("Enter text: ")
        print("Category:", run_pipeline(text, input_type="text"))
