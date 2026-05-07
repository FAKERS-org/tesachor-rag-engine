from fastapi import FastAPI, HTTPException, Depends
from dotenv import load_dotenv
load_dotenv()

from schemas import GenerateRequest, GenerateResponse, HealthResponse
from dependencies import get_provider
from providers.base import BaseLLMProvider
from exceptions import LLMAPIError, LLMError
from config import config

import traceback
import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tesachor RAG Engine - LLM Service",
    description="Multi-provider LLM generation service",
    version="1.0.0",
)

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    provider: BaseLLMProvider = Depends(get_provider),
):
    try:
        logger.info(f"Generating text with provider: {config.provider}, model: {config.model_name}")
        return await provider.generate(request)
    except LLMAPIError as e:
        logger.error(f"LLM API Error: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))
    except LLMError as e:
        logger.error(f"LLM Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in LLM service: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check(
    provider: BaseLLMProvider = Depends(get_provider)
):
    try:
        health = await provider.health_check()
        return HealthResponse(
            status=health.get("status", "healthy"),
            provider=config.provider,
            model=config.model_name
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            provider=config.provider,
            model=config.model_name
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)