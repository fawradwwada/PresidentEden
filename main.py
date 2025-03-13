from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load DistilGPT-2 model and tokenizer (smaller version of GPT-2)
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")  # Use distilgpt2 as the default
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # Run on CPU
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
