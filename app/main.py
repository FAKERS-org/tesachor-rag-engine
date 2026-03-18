from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.routes import router
from app.api.models import QueryRequest, QueryResponse, HealthCheck
from app.core.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStoreService
from app.services.cache import RedisCache
from app.config import settings

# Setup logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global service instances
rag_pipeline: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services"""
    global rag_pipeline
    
    logger.info("🚀 Starting RAG Engine Service...")
    
    # Initialize components
    vector_store = VectorStoreService()
    cache = RedisCache() if settings.ENABLE_CACHE else None
    
    # Build RAG pipeline (dependency injection)
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=None,  # Will initialize in pipeline
        embedding_client=vector_store.embeddings,
        cache_client=cache,
        reranker=None  # Add cross-encoder here
    )
    
    logger.info("RAG Engine ready")
    yield
    
    # Cleanup
    logger.info("Shutting down RAG Engine...")
    if cache:
        await cache.close()

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
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "vector_store": rag_pipeline.vector_store.get_collection_stats() if rag_pipeline else None
    }

# Make pipeline available to routes
def get_rag_pipeline():
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline