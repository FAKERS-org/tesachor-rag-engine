from pydantic import BaseModel, Field
from typing import List, Dict, Any

class EmbedRequest(BaseModel):
    # validate that sentences is a list of strings
    sentences: List[str] = Field(..., min_items=1, max_items=100)

    class Config:
        json_schema_extra = {
            "example": {
                "sentences": [
                    "The cat sat on the mat.",
                    "The dog barked loudly."
                ]
            }
        }

class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., min_items=1) # validate that embeddings is a list of lists of floats
    provider: str = Field(..., description="Provider used for embeddings")
    model: str = Field(..., description="Model used for embeddings")

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]
                ],
                "provider": "cohere",
                "model": "small"
            }
        }


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    details: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "provider": "cohere",
                "model": "small",
                "details": {
                    "healthy": True,
                    "latency": "10ms"
                }
            }
        }
