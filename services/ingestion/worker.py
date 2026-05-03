from celery import Celery
import asyncio
from psycopg2.extras import execute_values

app = Celery('ingestion', broker='redis://redis:6379')

@app.task
def bulk_ingest_documents(documents: list, version_hash: str):
    
    # chunk all document
    all_chunks = []
    for doc in documents:
        chunks = chunk_documents(doc['content'])
        all_chunks.extend([(chunk, doc['metadata']) for chunk in chunks])
        
    # batch embedding
    embeddings = []
    for batch in chunks_batch(all_chunks, size=100):
        batch_embeddings = embed_service.encode([c[0] for c in batch])
        embeddings.extend(batch_embeddings)
        
    # bulk insert to pgvector
    execute_values(conn.cursor(), """
            INSERT INTO documents (content, embedding, metadata, version_hash)
            VALUES %s
        """, [(chunks, emb, meta, version_hash) for (chunks, meta), emb in zip(all_chunks, embeddings)])
    
    # log to mlflow
    mlflow.log_params({
        "chunk_size": 512,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 100,
        "total_chunks": len(all_chunks),
        "version_hash": version_hash
    })