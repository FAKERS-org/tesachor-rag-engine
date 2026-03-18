from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Service
    SERVICE_NAME: str = "tesachor-rag-engine-v1"
    PORT: int = 8001

    # HuggingFace Inference API
    HF_API_TOKEN: Optional[str] = None
    # Featherless/OpenAI-compatible endpoint token
    FEATHERLESS_API_KEY: Optional[str] = None

    # Model Selection
    # Options: "BAAI/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    # LLM: Use shorthand or full model ID
    # "mistral-7b", "zephyr-7b", "gemma-2b", "microsoft/phi-2", etc.
    LLM_MODEL: str = "mistral-7b"

    # Or use dedicated endpoints (faster, no cold starts)
    HF_EMBEDDING_ENDPOINT: Optional[str] = None
    HF_LLM_ENDPOINT: Optional[str] = None
    # Inference provider suffix (e.g. featherless-ai)
    HF_LLM_PROVIDER: Optional[str] = None

    # Generation settings
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512

    # Vector DB
    VECTOR_DB_PATH: str = "./data/vector_db"
    COLLECTION_NAME: str = "tesachor_knowledge"
    TOP_K_RETRIEVAL: int = 5

    # Retrieval controls
    HYBRID_SEARCH: bool = False
    RERANK_ENABLED: bool = False
    SIMILARITY_THRESHOLD: float = 0.0

    # Performance
    ENABLE_CACHE: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600

    # Request batching (important for HF API efficiency)
    EMBEDDING_BATCH_SIZE: int = 32  # HF API supports batching
    STARTUP_SELF_TEST: bool = False
    DATASET_DIR: str = "./data/original"
    ALT_DATASET_DIR: str = "./data/transformed"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
