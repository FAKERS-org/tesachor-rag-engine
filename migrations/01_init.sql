-- ==========================================================
-- Phase 1: Foundation - Database & Schema Initialization
-- ==========================================================

-- 1. Create additional databases for our MLOps ecosystem.
-- MLflow needs its own database to store experiment metadata.
-- LakeFS (if configured for Postgres backend) needs its own database.
-- Note: In a production script, we'd check if they exist first, 
-- but for initialization, simple CREATE commands work.
CREATE DATABASE mlflow;
CREATE DATABASE lakefs;

-- 2. Setup the 'rag' database (default database)
-- Enable the pgvector extension. This is what transforms PostgreSQL 
-- from a standard relational DB into a high-performance Vector Database.
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. Create the documents table.
-- This table is designed for "Traceable RAG":
-- - content: The actual text chunk.
-- - embedding: The 384D vector (optimized for all-MiniLM-L6-v2).
-- - embedding: The 1024D vector (optimized for the specified embedding model, Cohere).
-- - metadata: Flexible JSONB to store source info, headers, or page numbers.
-- - version_hash: The link to LakeFS, allowing us to know exactly which 
--   version of a file generated this chunk.
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    metadata JSONB DEFAULT '{}',
    version_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create an HNSW (Hierarchical Navigable Small World) Index.
-- In production RAG, a "sequential scan" (checking every row) is too slow.
-- HNSW creates a graph-based index that allows us to find the 'Nearest Neighbors' 
-- in logarithmic time. 
-- 'vector_cosine_ops' is used because Cosine Similarity is the industry standard 
-- for comparing text embeddings.
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
