import os
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

from providers.base import BaseLLMProvider
from schemas import GenerateRequest, GenerateResponse
from config import LLMConfig
from exceptions import LLMAPIError, ProviderNotConfiguredError, LLMError

class OpenRouterLLMProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base or "https://openrouter.ai/api/v1"
        )
        self.model_name = config.model_name
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            extra_body = {}

            # reasoning capability
            if request.enable_reasoning:
                extra_body["reasoning"] = {"enabled": True}

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                extra_body=extra_body,
                extra_headers={
                    "HTTP-Referer": "https://github.com/tesachor/tesachor-rag-engine",
                    "X-Title": "Tesachor RAG Engine",
                }
            )
            
            # extract content
            message = response.choices[0].message
            content = message.content or ""
            
            # extract reasoning - OpenRouter can put it in different places
            reasoning = None
            if hasattr(message, "reasoning_details"):
                reasoning = message.reasoning_details
            elif hasattr(message, "reasoning"):
                reasoning = message.reasoning
            
            # track token usage
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                
                # reasoning tokens when available
                if hasattr(response.usage, "reasoning_tokens"):
                    usage["reasoning_tokens"] = response.usage.reasoning_tokens
            
            return GenerateResponse(
                text=content,
                reasoning=reasoning,
                model=self.model_name,
                provider=self.config.provider or "openrouter",
                usage=usage if usage else None,
            )
        
        except Exception as e:
            raise LLMAPIError(f"OpenRouter API error: {str(e)}")
        
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "provider": self.config.provider or "openrouter",
        }