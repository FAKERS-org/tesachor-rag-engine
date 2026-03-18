"""
Service factory: Choose between local models or HF Inference API
"""

from app.config import settings
from app.services.hf_inference_embeddings import HFInferenceEmbeddings, HFInferenceEndpointEmbeddings
from app.services.hf_inference_llm import HFInferenceLLM, HFInferenceEndpointLLM

def get_embeddings_service():
    """Factory: Return appropriate embedding service"""

    # Priority 1: Dedicated endpoint (if configured)
    if settings.HF_EMBEDDING_ENDPOINT:
        return HFInferenceEndpointEmbeddings(
            endpoint_url=settings.HF_EMBEDDING_ENDPOINT,
            api_token=settings.HF_API_TOKEN
        )

    # Priority 2: HF Inference API (serverless)
    if settings.HF_API_TOKEN:
        return HFInferenceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            api_token=settings.HF_API_TOKEN
        )

    # Priority 3: Local models (fallback)
    from app.services.hf_embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

def get_llm_service():
    """Factory: Return appropriate LLM service"""

    # Priority 1: Dedicated endpoint
    if settings.HF_LLM_ENDPOINT:
        return HFInferenceEndpointLLM(
            endpoint_url=settings.HF_LLM_ENDPOINT,
            api_token=settings.HF_API_TOKEN
        )

    # Priority 2: HF Inference API
    if settings.HF_API_TOKEN:
        return HFInferenceLLM(
            model_id=settings.LLM_MODEL,
            api_token=settings.HF_API_TOKEN
        )

    # Priority 3: Local
    from app.services.hf_llm import HuggingFaceLLM
    return HuggingFaceLLM(model_name=settings.LLM_MODEL)
