from typing import List
import cohere
from .base import BaseEmbeddingProvider
from ..config import EmbeddingConfig
from ..exceptions import ProviderNotConfiguredError, EmbeddingAPIError # <- check api from providerr

class CohereEmbeddingProvider(BaseEmbeddingProvider):
    
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        # check api provision
        if not config.api_key:
            raise ProviderNotConfiguredError("Cohere API key is not configured.")
        
        # init cohere client
        self.client = cohere.Client(config.api_key)
    
    async def embed(self, sentences: List[str]) -> List[List[float]]:
        try:
            # 
            response = await self.client.embed(
                texts=sentences,
                model=self.config.model_name,
                
            )
            return response.embeddings
        except Exception as e:
            raise EmbeddingAPIError(f"Cohere embedding API failed: {e}")

    async def health_check(self) -> dict:
        try:
            # simple test call or check client
            return {
                "healthy": True,
                "provider": "cohere",
                "model": self.config.model_name
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }