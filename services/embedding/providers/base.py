from abc import ABC, abstractmethod
from typing import List
from config import EmbeddingConfig

class BaseEmbeddingProvider(ABC): # abstarct super parent class
    
    def __init__(self, config: EmbeddingConfig) -> EmbeddingConfig:
        self.config = config
        
    @abstractmethod
    async def embed(self, sentences: List[str]) -> List[List[float]]:
        # embeddings for list of texts
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        # check provider health
        pass
    
        