from typing import Dict, Type

from providers.base import BaseEmbeddingProvider
from providers.local import LocalEmbeddingProvider
from providers.cohere import CohereEmbeddingProvider
# from openai import OpenAIProvider
# from gemini import GeminiProvider
from config import config
from exceptions import ProviderNotConfiguredError

# map
PROVIDER_MAP: Dict[str, Type[BaseEmbeddingProvider]] = {
    "local": LocalEmbeddingProvider,
    "cohere": CohereEmbeddingProvider,
    # "openai": OpenAIProvider,
    # "gemini": GeminiProvider,
}

def get_embedding_provider() -> BaseEmbeddingProvider:
    
    # check if provider is configured
    provider_class = PROVIDER_MAP.get(config.provider)

    # if not, raiser error and show alternatives
    if not provider_class:
        raise ProviderNotConfiguredError(
            f"Unknown Embedding Provider: {config.provider}. Available: {list(PROVIDER_MAP.keys())}"
        )
        
    # return instance
    return provider_class(config=config)