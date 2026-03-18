# from pydantic_settings import BaseSettings
# from functools import lru_cache
# from typing import Optional

# class Settings(BaseSettings):
#     # Service(v1)
#     SERVICE_NAME: str = "phnompenh-rag-engine"
#     PORT: int = 8001
#     DEBUG: bool = False
    
#     # OpenAI
#     OPENAI_API_KEY: str
#     EMBEDDING_MODEL: str = "text-embedding-3-small"
#     LLM_MODEL: str = "gpt-4"
#     LLM_TEMPERATURE: float = 0.7
#     LLM_MAX_TOKENS: int = 1000
    
#     # Vector DB
#     VECTOR_DB_PATH: str = "./data/vector_db"
#     COLLECTION_NAME: str = "phnompenh_attractions"
#     TOP_K_RETRIEVAL: int = 5
    
#     # Retrieval Settings (AI Engineering tuning)
#     SIMILARITY_THRESHOLD: float = 0.7  # Filter low relevance
#     RERANK_ENABLED: bool = True        # Cross-encoder reranking
#     HYBRID_SEARCH: bool = True         # Vector + Keyword
    
#     # Performance
#     ENABLE_CACHE: bool = True
#     REDIS_URL: Optional[str] = "redis://localhost:6379/0"
#     CACHE_TTL: int = 3600  # 1 hour
    
#     # Monitoring
#     LOG_LEVEL: str = "INFO"
    
#     class Config:
#         env_file = ".env"

# @lru_cache()
# def get_settings() -> Settings:
#     return Settings()

# settings = get_settings()

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Service
    SERVICE_NAME: str = "phnompenh-rag-engine"
    PORT: int = 8001
    
    # HuggingFace Models (FREE, local)
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"  # 768 dim, fast, good quality
    LLM_MODEL: str = "HuggingFaceH4/zephyr-7b-beta"   # 7B params, chat-tuned
    
    # Or use these for CPU-only:
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LLM_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512  # Shorter for local models (faster)
    
    # Vector DB
    VECTOR_DB_PATH: str = "./data/vector_db"
    TOP_K_RETRIEVAL: int = 5
    
    # Performance
    BATCH_SIZE: int = 32  # For embedding generation
    ENABLE_CACHE: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()