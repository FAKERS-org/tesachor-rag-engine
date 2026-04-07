"""
Vector Database Abstraction
Supports pgvector (PostgreSQL with pgvector extension)
"""

import asyncio
from typing import List, Tuple, Optional
from langchain_postgres import PGVector
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = PGVector(
            connection=settings.POSTGRES_URL,
            embeddings=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            use_jsonb=True,
        )
        logger.info(f"Connected to pgvector at {settings.POSTGRES_URL.split('@')[-1]}")

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Tuple]:
        """Async similarity search with relevance scores using thread executor"""
        # We use asyncio.to_thread because langchain-postgres requires async_mode=True 
        # for true async calls, which would require an async driver/pool setup.
        return await asyncio.to_thread(
            self.store.similarity_search_with_score, 
            query, 
            k=k, 
            filter=filter
        )

    async def add_documents(self, texts: List[str], metadatas: List[dict]):
        """Add new documents to vector store using thread executor"""
        await asyncio.to_thread(self.store.add_texts, texts=texts, metadatas=metadatas)
        logger.info(f"Added {len(texts)} documents to vector store")

    async def reset_collection(self):
        """Drop all existing vectors in the current collection."""
        await asyncio.to_thread(self.store.drop_tables)
        self.store = PGVector(
            connection=settings.POSTGRES_URL,
            embeddings=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            use_jsonb=True,
        )
        logger.info("Reset pgvector collection")

    def get_collection_stats(self) -> dict:
        """Get statistics about stored data"""
        # PGVector doesn't have a direct .count() like Chroma, 
        # but we can try to estimate or return basic info.
        return {
            "collection_name": settings.COLLECTION_NAME,
            "embedding_model": settings.EMBEDDING_MODEL,
            "db_type": "pgvector"
        }
