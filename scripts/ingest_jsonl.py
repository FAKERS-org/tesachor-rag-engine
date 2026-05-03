import json
import requests
import time

API_URL = "http://localhost:8000/ingest"

def ingest_data(file_path: str):
    print(f"Reading data from {file_path}...")
    
    success_count = 0
    error_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                payload = {
                    "content": data["content"],
                    "metadata": data.get("metadata", {}),
                    "version_hash": "v1-initial-load"
                }
                
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                success_count += 1
                
                if success_count % 100 == 0:
                    print(f"Queued {success_count} documents for ingestion...")
                    
            except Exception as e:
                print(f"Failed to ingest document: {e}")
                error_count += 1
                
    print(f"Ingestion queueing complete! Successfully queued: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    ingest_data("data/rag_documents.jsonl")