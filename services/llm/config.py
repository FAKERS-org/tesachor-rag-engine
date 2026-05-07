import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    provider: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openrouter").lower(),
            model_name=os.getenv("LLM_MODEL_NAME", "openrouter/free"),
            api_key=os.getenv("LLM_API_KEY"),
            api_base=os.getenv("LLM_API_BASE_URL", "https://openrouter.ai/api/v1")
        )
        
config = LLMConfig.from_env()