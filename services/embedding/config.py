import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingConfig: # whole config for embedding client
    provider: str
    model_name: str
    api_key: Optional[str] = None
    device: str = "cpu" # default to cpu, but prefer gpu if avaiable
    
    @classmethod
    def from_env(cls):
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "cohere").lower(),
            model_name=os.getenv("EMBEDDING_MODEL_NAME", ""),
            api_key=os.getenv("EMBEDDING_API_KEY"),
            device=os.getenv("EMBEDDING_DEVICE", "cuda" if cls._has_cuda() else "cpu")
        )

        
    @staticmethod # cuda checker
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

# export
config = EmbeddingConfig.from_env()