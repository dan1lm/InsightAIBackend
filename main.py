from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

try:
    # Set HF token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")
    
    generator = pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None

class HealthData(BaseModel):
    healthData: str

@app.post("/generate-insight")
async def generate_insight(data: HealthData):
    if not generator:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        prompt = f"""[INST]Based on the following health data, provide a brief, meaningful insight in 2-3 sentences. Focus on identifying patterns and suggesting one actionable improvement:

{data.healthData}[/INST]"""
        
        response = generator(
            prompt,
            max_length=256,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,
        )[0]['generated_text']
        
        # Extract just the response part after the prompt
        insight = response.split("[/INST]")[-1].strip()
        
        return {"insight": insight}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": generator is not None}