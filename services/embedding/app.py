from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import os
from typing import List

# 1. Initialize FastAPI
app = FastAPI(title="Tesachor Embedding Service")

# 2. Model Configuration
# We fetch the model name from environment variables, defaulting to the 
# efficient all-MiniLM-L6-v2 (384 dimensions).
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Check if a GPU is available. In production, using a GPU for embeddings 
# can be 10x-100x faster than a CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Loading model: {EMBEDDING_MODEL_NAME} on {DEVICE} ---")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

# 3. Define Request/Response Schemas
class EmbedRequest(BaseModel):
    sentences: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

# 4. The /encode endpoint
@app.post("/encode", response_model=EmbedResponse)
async def encode_sentences(request: EmbedRequest):
    """
    Converts a list of strings into a list of vector embeddings.
    """
    try:
        if not request.sentences:
            return EmbedResponse(embeddings=[])
        
        # We use model.encode to generate vectors.
        # convert_to_tensor=False returns a list of numpy arrays, 
        # which we then convert to standard Python lists for JSON serialization.
        embeddings = model.encode(request.sentences, convert_to_tensor=False)
        
        return EmbedResponse(embeddings=embeddings.tolist())
    
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME, "device": DEVICE}
