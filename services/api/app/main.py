from fastapi import FastAPI, HTTPException, BackgroundTasks
from celery import Celery
import os

from .utils import get_embedding, vector_search
from .schemas import QueryRequest, IngestRequest

# init fasdtapi
app = FastAPI(title="Tesachor RAG API")

# connect to celeery for ingetion
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery('ingestion', broker=REDIS_URL)

# query endpoit
@app.post("/query")
async def query_rag(request: QueryRequest):
    # searc for relevant chunks
    try:
        # question -> vector
        vector = get_embedding(request.question)

        # search db
        results = vector_search(vector, limit=request.top_k)
        
        return {"results" : results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ingestion enfpoit
@app.post("/ingest")
async def trigger_ingestion(request: IngestRequest):
    # push doc to bg workers for proccessing
    
    # we send the task to celery for the workers to handle chunking and embedding in the bg
    celery_app.send_task(
        "ingest_document_task",
        args=[request.content, request.metadata, request.version_hash],
    )
    return {"status" : "accepted", "message" : "Documents queued for processing"}

# root api
@app.get("/")
async def root():
    return {"message" : "Tesachor RAG API is online"}

# health
@app.get("/health")
async def health():
    return {"message" : "Tesachor RAG API is Healthy"}