import pandas as pd
import google.generativeai as genai
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_PLUGIN_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


API_KEY = "AIzaSyBmC_o1JQL0q7CzGMEKEQS7M-pO34_lx88"
genai.configure(api_key=API_KEY)


SYSTEM_PROMPT = """
You are MindSeek, a licensed and kind mental health therapist.
Always respond gently and constructively. Respond with empathy.
- Use natural, conversational English with contractions (I'm, you're, let’s)
- Occasionally include thoughtful pauses: "Hmm, let’s think about that..."
- Talk less, listen more.
- Do not mention medications or diagnoses.
- If the topic is out of scope, politely say you cannot answer it.
"""

dataset_path = "mhqa_cleaned.csv"

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Dataset not found at {dataset_path}. Continuing without examples.")
    df = None

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

# --- 6. GENERATE RESPONSE USING GEMINI ---
def generate_gemini_response(user_input, dataframe):
    prompt = build_prompt_with_context(user_input, dataframe)

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        print(response.text)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
   

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace * with your frontend URL like "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    result = generate_gemini_response(user_input, df)
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)