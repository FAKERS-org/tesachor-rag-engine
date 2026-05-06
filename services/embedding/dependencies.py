from fastapi import Request
from .providers import get_embedding_provider
from .providers.base import BaseEmbeddingProvider

# dependency
async def get_provider(request: Request) -> BaseEmbeddingProvider:
    
    # lazy load provider and store in app state
    if not hasattr(request.app.state, "embedding_provider"):
        # store provider instance in app state for reuse across requests
        request.app.state.embedding_provider = get_embedding_provider()
    
    return request.app.state.embedding_provider
