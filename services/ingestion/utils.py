import os
import requests
import psycopg2
from psycopg2.extras import execute_values
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Database Connection Helper
def get_db_connection():
    # Fetch DSN from environment variables
    dsn = os.getenv("PGVECTOR_DSN", "postgresql://user:pass@postgres:5432/rag")
    return psycopg2.connect(dsn)

# 2. Embedding Service Client
def get_embeddings(texts: list[str]):
    url = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding:8080")
    response = requests.post(f"{url}/encode", json={"sentences": texts})
    response.raise_for_status()
    return response.json()["embeddings"]

# 3. Intelligent Chunking
# RecursiveCharacterTextSplitter is the industry standard for RAG.
# It tries to split by paragraphs first, then sentences, then words,
# ensuring we don't cut a thought in half.
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# 4. Bulk Ingestion Logic
def bulk_insert_documents(conn, data_list):
    """
    Uses psycopg2.extras.execute_values for high-speed batch insertion.
    data_list should be a list of tuples: (content, embedding, metadata, version_hash)
    """
    query = """
        INSERT INTO documents (content, embedding, metadata, version_hash)
        VALUES %s
    """
    with conn.cursor() as cur:
        execute_values(cur, query, data_list)
    conn.commit()
