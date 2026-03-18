import json
from app.services.hf_embeddings import HuggingFaceEmbeddings
from app.services.vector_store import VectorStoreService
from app.config import settings

def build_from_transformed():
    # Initialize services
    embeddings = HuggingFaceEmbeddings()
    vector_store = VectorStoreService(embeddings=embeddings)  # Pass custom embeddings
    
    # Load transformed documents
    documents = []
    with open("data/rag_documents.jsonl", 'r') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
    
    print(f"Loading {len(documents)} documents...")
    
    # Batch embed and store
    texts = [d['content'] for d in documents]
    metadatas = [d['metadata'] for d in documents]
    
    # Add IDs
    for i, meta in enumerate(metadatas):
        meta['doc_id'] = f"doc_{i}"
    
    vector_store.add_documents(texts, metadatas)
    print("Vector store built with HuggingFace embeddings!")

if __name__ == "__main__":
    build_from_transformed()