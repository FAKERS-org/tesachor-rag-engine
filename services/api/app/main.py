from fastapi import FastAPI, HTTPException
from celery import Celery
import os

from .utils import get_embedding, vector_search, generate_answer
from .schemas import QueryRequest, IngestRequest


# init fasdtapi
app = FastAPI(title="Tesachor RAG API")

# connect to celeery for ingetion
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery('ingestion', broker=REDIS_URL)

# query endpoint
# DESCRIPTION: We take the content from each database result and glue it together into one big string (context). 
# We then wrap that context and the user's question into a strict instruction set (prompt).
# We send it to Gemini, and then return the final answer.
@app.post("/query")
async def query_rag(request: QueryRequest):
    # searc for relevant chunks
    try:
        # question -> vector
        vector = get_embedding(request.question)
        
        # vector -> search db
        results = vector_search(vector, limit=request.top_k)
        
        # prepare the context by combining all retreived chunks
        # but keep them seperated
        context = "\n\n".join([doc["content"] for doc in results])
        
        # build prompt for llm to behave
        prompt = f"""You are a helpful assistant for the Tesachor RAG system.
Answer the user's question using ONLY the provided reteived documents as context.
If the answer is not found in the provided context, say "I cannot answer this based on the provided doccuments. Please Consider Tesachor Team for more info."

Context:
{context}

Question: {request.question}
Answer:"""

        # generate the answer by sending the prompt to the llm client
        answer = generate_answer(prompt)
        
        # return both the generated answer and
        # the source documents for transparency (citations) and debugging
        return {
            "answer": answer,
            "results" : results,
        }
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