import os
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from google import genai

# Database Connection with RealDictCursor
# RealDictCursor makes the database results look like Python dictionaries
# (e.g., result['content']) instead of tuples (result[0]), which is much 
# easier to work with in an API.
def get_db_connection():
    dsn = os.getenv("PGVECTOR_DSN", "postgresql://user:pass@localhost:5432/rag")
    return psycopg2.connect(dsn, cursor_factory=RealDictCursor)

# Embedding Client (same as ingestion worker)
def get_embedding(text: str):
    url = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080")
    response = requests.post(f"{url}/encode", json={"sentences": [text]})
    response.raise_for_status()
    # Return just the first embedding since we only sent one sentence
    return response.json()["embeddings"][0]

# The Core Vector Search Logic
def vector_search(query_vector: list[float], limit: int = 5):
    """
    Finds the nearest neighbors using Cosine Distance (<=>).
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Postgres Vector Math:
            # '<=>' calculates Cosine Distance.
            # 1 - distance = Cosine Similarity (where 1.0 is a perfect match).
            query = """
                SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(query, (query_vector, query_vector, limit))
            return cur.fetchall()
    finally:
        conn.close()

# llm genration client
def generate_answer(prompt: str):
    """
    Sends the prompt to the selected LLM provider.
    Now using the google-genai SDK.
    """
    
    provider = os.getenv("LLM_PROVIDER", "google")
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro-002")
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise ValueError("LLM_API_KEY not found")
    
    if provider == "google":
        # init the official google genai client
        client = genai.Client(api_key=api_key)
        
        # send the prompt to the model
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        
        return response.text
    
    else:
        raise NotImplementedError(f"Provider {provider} is not supported yet.")