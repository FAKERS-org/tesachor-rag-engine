from pydantic import BaseModel
from typing import Optional

# Schemas
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    content: str
    metadata: dict
    version_hash: str