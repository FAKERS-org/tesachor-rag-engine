import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
load_dotenv()

from providers.base import BaseLLMProvider
from schemas import GenerateRequest, GenerateResponse, HealthResponse

class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self):
        # use openai capability
        self.client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        )

class OpenAISDK():
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("LLM_SDK_API_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv()
        )