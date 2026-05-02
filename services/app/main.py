import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router
from app.api.v1.schemas import HealthCheck
from app.config import settings
from app.core.rag_pipeline import RAGPipeline
from app.services.factory import get_embeddings_service, get_llm_service
from app.services.vector_store import VectorStoreService

# Setup logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global service instance
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Engine with HF Inference...")

    # Initialize via factory
    embeddings = get_embeddings_service()
    llm = get_llm_service()

    # Optional startup self-test
    if settings.STARTUP_SELF_TEST:
        try:
            test_embed = await embeddings.aembed_query("Cambodia")
            logger.info(f"Embeddings: {len(test_embed)}d")

            test_response = await llm.ainvoke([
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Say 'HF Inference working'"}
            ])
            logger.info(f"LLM: {test_response.content[:50]}...")
        except Exception as e:
            logger.error(f"Model connectivity check failed: {e}")
            raise

    # Initialize pipeline
    global rag_pipeline
    vector_store = VectorStoreService(embeddings=embeddings)
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=llm,
        embedding_client=embeddings,
        cache_client=None
    )
    app.state.rag_pipeline = rag_pipeline

    logger.info("RAG Engine ready with HF Inference")
    yield

    # Cleanup
    logger.info("Shutting down...")

app = FastAPI(
    title="Phnom Penh RAG Engine",
    description="AI-powered retrieval service for tourism data",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for chat-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to chat-backend in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Service health and stats"""
    pipeline = getattr(app.state, "rag_pipeline", None)
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "llm_model": settings.LLM_MODEL,
        "vector_store": pipeline.vector_store.get_collection_stats() if pipeline else None
    }


def main():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0",
                port=settings.PORT, reload=False)
