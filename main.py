from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model (default: "mistralai/Mistral-7B-Instruct")
MODEL_NAME = os.getenv("MODEL_NAME", "tiiuae/falcon-7b-instruct")
generator = pipeline("text-generation", model=MODEL_NAME)
logger.info(f"Loaded model: {MODEL_NAME}")

class Prompt(BaseModel):
    text: str
    max_length: int = 100

@app.post("/generate")
async def generate_text(prompt: Prompt):
    logger.info(f"Received prompt: {prompt.text}")
    output = generator(prompt.text, max_length=prompt.max_length)
    generated_text = output[0]['generated_text']
    logger.info(f"Generated text: {generated_text}")
    return {"generated_text": generated_text}

@app.get("/")
def home():
    return {"message": "LLM API is running!"}

