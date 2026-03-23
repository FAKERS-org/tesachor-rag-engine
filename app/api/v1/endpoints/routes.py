from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.api.v1.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    RetrievalOnlyRequest
)
from app.core.rag_pipeline import RAGPipeline

router = APIRouter()


def get_rag_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Full RAG pipeline: Retrieve + Generate
    Main endpoint for chat-backend
    """
    try:
        result = await pipeline.query(
            question=request.question,
            filters=request.filters,
            conversation_context=request.conversation_context,
            top_k=request.top_k
        )

        return {
            "answer": result.answer,
            "sources": [
                {
                    "name": c.metadata.get("name"),
                    "type": c.source_type,
                    "location": c.metadata.get("location"),
                    "relevance_score": c.score
                }
                for c in result.retrieved_chunks
            ],
            "metrics": {
                "retrieval_time_ms": result.retrieval_time_ms,
                "generation_time_ms": result.generation_time_ms,
                "total_tokens": result.total_tokens,
                "confidence": result.confidence_score
            },
            "cached": False  # Would be set by cache layer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve-only")
async def retrieve_only(
    request: RetrievalOnlyRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Retrieval-only endpoint for debugging/testing
    Returns raw chunks without generation
    """
    chunks = await pipeline._retrieve(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k or 5
    )

    return {
        "query": request.query,
        "results": [
            {
                "content": c.content,
                "metadata": c.metadata,
                "score": c.score
            }
            for c in chunks
        ]
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_data(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Ingest new JSON data into vector store
    Runs in background to avoid blocking
    """
    # Validate JSON structure
    if not isinstance(request.data, list):
        raise HTTPException(
            status_code=400, detail="Data must be a list of objects")

    # Background task for processing
    background_tasks.add_task(
        process_ingestion,
        request.data,
        pipeline.vector_store
    )

    return {
        "status": "processing",
        "records_received": len(request.data),
        "estimated_time_seconds": len(request.data) * 2  # Rough estimate
    }


async def process_ingestion(data: list, vector_store):
    """Background processing: chunk and embed"""
    from app.utils.chunking import chunk_json_records

    # Chunk the data
    chunks = chunk_json_records(data)

    # Add to vector store
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vector_store.add_documents(texts, metadatas)
