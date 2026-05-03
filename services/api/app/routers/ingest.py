from celery.result import AsyncResult

@app.post("/ingest/bulk")
async def trigger_ingestion(files: list[UploadFile])
    # process backgound
    task = bulk_ingest_documents.delay(documents, version_hash="commi-123")
    return {
        "task_id": task.id,
        "status": "processing"
    }

@app.get("/ingest/status/{task_id}")
async def get_status(task_id: str):
    task = AsyncResult(task_id)
    return {
        "status": task.state,
        "progress": task.info
    }