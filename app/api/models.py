from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[Dict[str, Any]] = None
    conversation_context: Optional[List[Dict[str, str]]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the best temples to visit?",
                "filters": {"category": "Temple", "entry_fee": "Free"},
                "conversation_context": [
                    {"user": "Tell me about historical sites", "bot": "Phnom Penh has...", "topic": "history"}
                ]
            }
        }

class SourceInfo(BaseModel):
    name: str
    type: str
    location: Optional[str]
    relevance_score: float

class Metrics(BaseModel):
    retrieval_time_ms: float
    generation_time_ms: float
    total_tokens: int
    confidence: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    metrics: Metrics
    cached: bool

class RetrievalOnlyRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

class IngestRequest(BaseModel):
    data: List[Dict[str, Any]]
    source_name: Optional[str] = "manual_upload"

class IngestResponse(BaseModel):
    status: str
    records_received: int
    estimated_time_seconds: int

class HealthCheck(BaseModel):
    status: str
    service: str
    version: str
    vector_store: Optional[Dict]