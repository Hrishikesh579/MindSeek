from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os

# --- Gemini imports ---
import pandas as pd
import google.generativeai as genai
from model2_text.predict_text import predict_text as predict_emotion
from integration.gemini_integration import get_gemini_response

# --- Load dataset for context (optional) ---
dataset_path = "mhqa_cleaned.csv"
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    df = None

# --- FastAPI setup ---
app = FastAPI(title="MindSeek — Chat + Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API setup ---
genai.configure(api_key="AIzaSyBmC_o1JQL0q7CzGMEKEQS7M-pO34_lx88")

SYSTEM_PROMPT =  """
You are MindSeek, a licensed and kind mental health therapist.
Always respond gently and constructively. Respond with empathy.
- Use natural, conversational English with contractions (I'm, you're, let’s)
- Occasionally include thoughtful pauses: "Hmm, let’s think about that..."
- Talk less, listen more.
- Do not mention medications or diagnoses.
- If the topic is out of scope, politely say you cannot answer it.
"""
def build_prompt_with_context(user_input, dataframe, num_examples=3):
    prompt = SYSTEM_PROMPT + "\n\n"
    if dataframe is not None and not dataframe.empty:
        prompt += "--- EXAMPLES ---\n"
        examples = dataframe.sample(n=min(num_examples, len(dataframe)))
        for _, row in examples.iterrows():
            prompt += f"User: {row['input']}\nTherapist: {row['output']}\n\n"
        prompt += "--- END EXAMPLES ---\n\n"
    prompt += f"User: {user_input}\nTherapist:"
    return prompt

def generate_gemini_response(user_input):
    prompt = build_prompt_with_context(user_input, df)
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text


# ==============================
# 🧠 CHAT ENDPOINT
# ==============================
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    result = generate_gemini_response(user_input)
    return {"response": result}


# ==============================
# 🎭 EMOTION ENDPOINT
# ==============================
@app.post("/emotion")
async def analyze_emotion(request: Request):
    data = await request.json()
    text = data.get("text", None)

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    prediction = predict_emotion(text)
    explanation = get_gemini_response(prediction, "text")

    return JSONResponse({
        "emotion": prediction,
        "detail": explanation
    })


# ==============================
# ⚙️ RUN SERVER
# ==============================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        uvicorn.run("mindspace.backend.applli:app", host="127.0.0.1", port=8000, reload=True)
    else:
        print("Run with: python -m mindspace.backend.applli serve")


