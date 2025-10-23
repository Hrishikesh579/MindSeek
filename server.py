# integration/server.py
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from integration.run_pipeline import run_pipeline

app = FastAPI()

# Allow requests from any origin (for frontend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- TEXT ROUTE ----------
@app.post("/predict_text")
async def predict_text_endpoint(text: str = Form(...)):
    """
    Expects multipart/form-data input with a 'text' field.
    Example: FormData { text: "I am happy" }
    """
    result = run_pipeline(text, input_type="text")
    return {"prediction": result}


# ---------- AUDIO ROUTE ----------
@app.post("/predict_audio")
async def predict_audio_endpoint(audio: UploadFile = File(...)):
    """
    Expects multipart/form-data with an audio file field 'audio'.
    Example: FormData { audio: <.wav file> }
    """
    # Save temporarily
    temp_path = f"temp_{audio.filename}"
    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    result = run_pipeline(temp_path, input_type="audio")
    return {"prediction": result}


@app.get("/")
async def root():
    return {"message": "Mindseek API running successfully"}
