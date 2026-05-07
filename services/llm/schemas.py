from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful assistant for the Tesachor RAG system."
    temperature: Optional[float] = 0.7
    max_tokens: int = 1024
    enable_reasoning: bool = True

class GenerateResponse(BaseModel):
    text: str
    reasoning: Optional[Any] = None
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None  # track token usage

class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str