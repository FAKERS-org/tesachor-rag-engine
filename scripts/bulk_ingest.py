import asyncio
from pathlib import Path
from ingestion.worker import bulk_ingest_documents

async def migrate_existing_documents():
    # Load all your existing documents
    documents = []
    for file in Path("/data/documents").glob("*.pdf"):
        documents.append({
            "content": extract_text(file),
            "metadata": {"source": file.name, "version": "v1"}
        })
    
    # Use LakeFS to version them
    repo = lakefs.Repository("documents")
    with repo.commit("Initial bulk ingest", metadata={"timestamp": "2024-01-01"}) as commit:
        for doc in documents:
            commit.upload(doc['metadata']['source'], doc['content'])
        
        # Trigger the bulk ingestion with version hash
        task = bulk_ingest_documents.delay(
            documents, 
            version_hash=commit.id
        )
        print(f"Started task {task.id}")
        task.get(timeout=3600)  # Wait for completion
        
if __name__ == '__main__':
    asyncio.run(migrate_existing_documents())