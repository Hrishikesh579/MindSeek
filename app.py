from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from integration.run_pipeline import run_pipeline_audio, run_pipeline_text  # We'll create these wrapper functions
import uvicorn

app = FastAPI()

# Allow CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded audio temporarily
    contents = await file.read()
    with open("temp_input.wav", "wb") as f:
        f.write(contents)
    
    # Call the audio pipeline
    prediction = run_pipeline_audio("temp_input.wav")
    return {"prediction": prediction}

@app.post("/predict/text")
async def predict_text(text: str = Form(...)):
    prediction = run_pipeline_text(text)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
