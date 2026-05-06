from typing import List
from sentence_transformers import SentenceTransformer
import torch

from providers.base import BaseEmbeddingProvider
from config import EmbeddingConfig
from exceptions import EmbeddingProviderError

# Local embedding provider (so lazy and bold)
class LocalEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            self.device = config.device
            self.model = SentenceTransformer( # load model
                config.model_name,
                device=self.device
            )
        except Exception as e:
            raise EmbeddingProviderError(f"Cannot load local model: {e}")

    
    async def embed(self, sentences: List[str]) -> List[List[float]]:
        try:
            # run in thread pool
            import asyncio
            # embeddings processing
            embeddings = await asyncio.to_thread(
                self.model.encode,
                sentences,
                convert_to_tensor=False                
            )
            # return as list of embedding lists
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingProviderError(f"Local embedding failed: {e}")
    
    async def health_check(self) -> dict:
        return {
            "model": self.config.model_name,
            "device": self.device,
            "is_model_loaded": self.model is not None
        }