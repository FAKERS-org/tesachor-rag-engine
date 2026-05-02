"""
Service factory: Choose between local models or HF Inference API
"""

from app.config import settings
from app.services.embeddings import HFInferenceEmbeddings, HFInferenceEndpointEmbeddings
from app.services.llm import HFInferenceEndpointLLM, HFInferenceLLM


def _resolve_llm_api_token():
    """Select API token for the configured LLM provider."""
    provider = (settings.HF_LLM_PROVIDER or "").lower()
    endpoint = (settings.HF_LLM_ENDPOINT or "").lower()

    # Featherless endpoints are OpenAI-compatible and use their own key.
    if "featherless" in provider or "featherless" in endpoint:
        return settings.FEATHERLESS_API_KEY

    return settings.HF_API_TOKEN


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
    from app.services.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def get_llm_service():
    """Factory: Return appropriate LLM service"""

    llm_api_token = _resolve_llm_api_token()
    endpoint = (settings.HF_LLM_ENDPOINT or "").lower()

    # Priority 1: Dedicated endpoint
    if settings.HF_LLM_ENDPOINT:
        if "featherless" in endpoint and not llm_api_token:
            raise ValueError(
                "Missing FEATHERLESS_API_KEY for Featherless endpoint authentication."
            )
        return HFInferenceEndpointLLM(
            endpoint_url=settings.HF_LLM_ENDPOINT,
            api_token=llm_api_token
        )

    # Priority 2: HF Inference API
    if llm_api_token:
        return HFInferenceLLM(
            model_id=settings.LLM_MODEL,
            api_token=llm_api_token
        )

    # Priority 3: Local
    from app.services.llm import HuggingFaceLLM
    return HuggingFaceLLM(model_name=settings.LLM_MODEL)
