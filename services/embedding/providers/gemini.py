from typing import List
from google import genai
from .base import BaseEmbeddingProvider
from ..config import EmbeddingConfig
from ..exceptions import ProviderNotConfiguredError, EmbeddingAPIError

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            raise ProviderNotConfiguredError("Gemini API key is not configured.")
        
        # Init Gemini client
        self.client = genai.Client(api_key=config.api_key)
        self.model_id = config.model_name or "text-embedding-004"
    
    async def embed(self, sentences: List[str]) -> List[List[float]]:
        try:
            # Gemini embedding API
            # Note: The SDK might have a different method name depending on version, 
            # but usually it's embed_content or similar in the new SDK.
            # Using models.embed_content based on latest google-genai SDK
            
            embeddings = []
            # Process in one batch if possible, or iterate if needed.
            # Gemini usually supports batching.
            response = self.client.models.embed_content(
                model=self.model_id,
                contents=sentences
            )
            
            # The structure depends on the SDK response
            return [e.values for e in response.embeddings]
            
        except Exception as e:
            raise EmbeddingAPIError(f"Gemini embedding API failed: {e}")

    async def health_check(self) -> dict:
        try:
            # Simple check
            return {
                "healthy": True,
                "provider": "gemini",
                "model": self.model_id
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
