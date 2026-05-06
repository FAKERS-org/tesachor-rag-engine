from typing import List
import cohere

from providers.base import BaseEmbeddingProvider
from config import EmbeddingConfig
from exceptions import ProviderNotConfiguredError, EmbeddingAPIError # <- check api from providerr

class CohereEmbeddingProvider(BaseEmbeddingProvider):
    
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        # check api provision
        if not config.api_key:
            raise ProviderNotConfiguredError("Cohere API key is not configured. Set EMBEDDING_API_KEY var")
        
        # init cohere client
        self.client = cohere.AsyncClient(config.api_key)
    
    async def embed(self, sentences: List[str]) -> List[List[float]]:
        try:
            # 
            response = await self.client.embed(
                texts=sentences,
                model=self.config.model_name or "embed-english-v3.0",
                input_type="search_query",
                embedding_types=["float"]
                
            )
            return response.embeddings.float
        
        except Exception as e:
            raise EmbeddingAPIError(f"Cohere embedding API failed: {e}")

    async def health_check(self) -> dict:
        try:
            # simple test call or check client
            await self.client.token(text="Health check", model='Command')
            return {"healthy": True, "provider": "cohere",}
        except Exception as e:
            return {"healthy": False, "error": str(e)}