"""
HuggingFace Inference Providers client for embeddings.
Uses huggingface_hub.InferenceClient instead of legacy HTTP endpoints.
"""

import asyncio
from typing import List
import logging
from huggingface_hub import InferenceClient
from app.config import settings

logger = logging.getLogger(__name__)


class HFInferenceEmbeddings:
    def __init__(self, model_name: str = None, api_token: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.api_token = api_token or settings.HF_API_TOKEN
        self.client = InferenceClient(
            api_key=self.api_token,
            provider="hf-inference",
        )

        # Dimension mapping
        self.dimension = {
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-en-v1.5": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "intfloat/multilingual-e5-large": 1024,
        }.get(self.model_name, 768)

        logger.info(
            f"HF Inference Embeddings: {self.model_name} ({self.dimension}d)")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed single query"""
        results = await self.aembed_documents([text])
        return results[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # The feature_extraction method expects a single string, so we need to call it for each text
        embeddings = [
            self.client.feature_extraction(text=text, model=self.model_name)
            for text in texts
        ]
        return self._normalize_embeddings(embeddings)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _normalize_embeddings(self, embeddings):
        import numpy as np
        embeddings = np.array(embeddings)
        if len(embeddings.shape) > 2:
            embeddings = embeddings.mean(axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        normalized = embeddings / norms
        return normalized.tolist()


class HFInferenceEndpointEmbeddings:
    """Dedicated HF Inference Endpoint for Embeddings"""

    def __init__(self, endpoint_url: str, api_token: str = None):
        self.api_url = endpoint_url
        self.api_token = api_token or settings.HF_API_TOKEN
        self.client = InferenceClient(
            api_key=self.api_token,
            base_url=self.api_url,
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        results = await self.aembed_documents([text])
        return results[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # The feature_extraction method expects a single string, so we need to call it for each text
        import numpy as np
        embeddings = [
            self.client.feature_extraction(text=text)
            for text in texts
        ]
        embeddings = np.array(embeddings)
        if len(embeddings.shape) > 2:
            embeddings = embeddings.mean(axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        normalized = embeddings / norms
        return normalized.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
