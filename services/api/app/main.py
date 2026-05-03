from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI

app = FastAPI()

# Automatic metrics
Instrumentator().instrument(app).expose(app)

# Manual metrics for RAG-specific monitoring
from prometheus_client import Histogram, Counter

query_latency = Histogram('rag_query_latency_seconds', 'Query processing time')
retrieval_precision = Histogram('rag_retrieval_precision', 'Precision@K')

@app.post("/query")
@query_latency.time()
async def query(request: QueryRequest):
    start_time = time.time()
    
    # Your existing retrieval logic
    results = retrieve_and_generate(request.query)
    
    # Log to MLflow for later evaluation
    mlflow.log_metrics({
        "latency_ms": (time.time() - start_time) * 1000,
        "retrieved_chunks": len(results['contexts']),
        "query_length": len(request.query)
    })
    
    # Store for offline evaluation
    await store_to_mongodb({
        "query": request.query,
        "response": results['answer'],
        "contexts": results['contexts'],
        "timestamp": datetime.utcnow()
    })
    
    return results