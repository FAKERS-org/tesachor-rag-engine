"""
HuggingFace Local Embeddings (Free, Private, Offline-capable)
Uses sentence-transformers for production-quality embeddings
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class HuggingFaceEmbeddings:
    """
    Local embedding model wrapper
    Recommended: "BAAI/bge-large-en-v1.5" (best quality/size ratio)
    Alternatives: "sentence-transformers/all-MiniLM-L6-v2" (faster, smaller)
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents (batch processing)"""
        # Normalize for cosine similarity
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        embedding = self.model.encode(
            text, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    # Async wrappers for FastAPI compatibility
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # Run in thread pool to not block event loop
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

# Model recommendations by use case
EMBEDDING_MODELS = {
    "quality": "BAAI/bge-large-en-v1.5",      # 1024 dim, best accuracy
    "balanced": "BAAI/bge-base-en-v1.5",    # 768 dim, good speed/quality
    "speed": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dim, fastest
    "multilingual": "intfloat/multilingual-e5-large"    # If you have Khmer text
}