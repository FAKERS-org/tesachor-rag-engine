import os
import json
from celery import Celery
import mlflow

from utils import get_db_connection, get_embeddings, chunk_text, bulk_insert_documents

# init celery
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
app = Celery('ingestion', broker=REDIS_URL)

# init mlflow
# Configure MLflow to talk to our server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RAG_Ingestion")

@app.task(name='ingest_document_task')
def ingest_document_task(content: str, metadata: dict, version_hash: str):
    # the bg task processing a single doc
    print(f"--- Processing doc: {metadata.get('title', 'unknown')} ---")
    
    # chunk the doc
    chunks = chunk_text(content)

    # get embedding in 1 batch
    embeddings = get_embeddings(chunks)

    # prep for bulk insert
    data_to_insert = []
    for chunk, embedding in zip(chunks, embeddings):
        data_to_insert.append(
            (
                chunk,
                embedding,
                json.dumps(metadata),
                version_hash
            )
        )
        
    # log the ingest to mlflow
    with mlflow.start_run():
        mlflow.log_param("chunk_size", 500)
        mlflow.log_param("chunk_overlap", 50)
        mlflow.log_param("model", "all-MiniLM-L6-v2")
        mlflow.log_param("version_hash", version_hash)

        # log the number id chuks created
        mlflow.log_metric("num_chunks", len(chunks))
                
    # load to db
    conn = get_db_connection()
    try:
        bulk_insert_documents(conn, data_to_insert)
        print(f"--- {len(chunks)} ingested ---")
    finally:
        conn.close()
    return {"status" : "success", "chunks": len(chunks)}