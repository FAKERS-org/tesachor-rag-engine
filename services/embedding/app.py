from fastapi import FastAPI, HTTPException, Depends
from dotenv import load_dotenv
load_dotenv()

from schemas import EmbedRequest, EmbedResponse, HealthResponse
from dependencies import get_provider
from providers.base import BaseEmbeddingProvider
from exceptions import EmbeddingProviderError, EmbeddingAPIError
from config import config

app = FastAPI(
    title="Tesachor RAG Engine - Embedding Service",
    description="Multi provider embedding services",
    version="1.1.0",
)

@app.post("/encode", response_model=EmbedResponse)
async def encode_sentences(
    request: EmbedRequest,
    provider: BaseEmbeddingProvider = Depends(get_provider),
):
    try:
        # generate embeddings
        embeddings = await provider.embed(request.sentences)
        
        # return response with provider and model info
        return EmbedResponse(
            embeddings=embeddings,
            provider=config.provider,
            model=config.model_name
        )
        
    except EmbeddingAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    except EmbeddingProviderError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check(
    provider: BaseEmbeddingProvider = Depends(get_provider)
):
    try:
        # get health
        provider_health = await provider.health_check()
        
        # return
        return HealthResponse(
            status="Healthy" if provider_health.get("healthy", False) else "Unhealthy",
            provider=config.provider,
            model=config.model_name,
            details=provider_health
        )
        
    # if error occurs
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            provider=config.provider,
            model=config.model_name,
            details={"error": str(e)}
        )
        
@app.on_event("startup")
async def startup_event():
    try:
        # test init provider
        from providers import get_embedding_provider
        test_provider = get_embedding_provider()
        
        # logs
        print(f"Provider '{config.provider}' initialized successfully")
        print(f"Model: {config.model_name}")
        
    except Exception as e:
        print(f"Error initializing provider: {str(e)}")