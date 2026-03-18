"""
Vector Database Abstraction
Supports Chroma (local) or Weaviate (production)
"""

from typing import List, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        self.store = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME
        )
        logger.info(f"Loaded vector store from {settings.VECTOR_DB_PATH}")
    
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Tuple]:
        """Async similarity search with relevance scores"""
        return await self.store.asimilarity_search_with_score(query, k=k, filter=filter)
    
    def add_documents(self, texts: List[str], metadatas: List[dict]):
        """Add new documents to vector store"""
        self.store.add_texts(texts=texts, metadatas=metadatas)
        self.store.persist()
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about stored data"""
        return {
            "document_count": self.store._collection.count(),
            "collection_name": settings.COLLECTION_NAME,
            "embedding_model": settings.EMBEDDING_MODEL
        }