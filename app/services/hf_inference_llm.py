"""
HuggingFace Inference Providers client for text generation.
"""

import asyncio
from typing import List, Dict
import logging
from huggingface_hub import InferenceClient
from app.config import settings

logger = logging.getLogger(__name__)

class HFInferenceLLM:
    """
    HF Inference API for conversational LLM
    Supports: Mistral-7B, Zephyr-7B, Llama-2-7B/13B, Gemma, etc.
    """
    
    # Free tier models (no dedicated endpoint needed)
    FREE_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "gemma-2b": "google/gemma-2b-it",
        "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",  # Gated, needs auth
        "phi-2": "microsoft/phi-2",
    }
    
    def __init__(self, model_id: str = None, api_token: str = None):
        self.model_id = model_id or settings.LLM_MODEL
        self.api_token = api_token or settings.HF_API_TOKEN
        
        # Resolve shorthand to full model ID
        if self.model_id in self.FREE_MODELS:
            self.model_id = self.FREE_MODELS[self.model_id]
        
        self.client = InferenceClient(
            api_key=self.api_token,
            provider="hf-inference",
        )
        
        logger.info(f"HF Inference LLM: {self.model_id}")
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for specific model
        Each model has different chat template!
        """
        # Extract system and user
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        user_prompt = "\n".join(user_msgs)
        
        # Mistral / Zephyr format
        model_id_lower = self.model_id.lower()

        if any(x in model_id_lower for x in ["mistral", "zephyr", "mixtral"]):
            if system:
                return f"<s>[INST] {system}\n\n{user_prompt} [/INST]"
            return f"<s>[INST] {user_prompt} [/INST]"
        
        # Llama-2 format
        elif "llama-2" in self.model_id.lower():
            if system:
                return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_prompt} [/INST]"
            return f"[INST] {user_prompt} [/INST]"
        
        # Gemma format
        elif "gemma" in self.model_id.lower():
            return f"<start_of_turn>user\n{system}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Generic
        else:
            return f"System: {system}\nUser: {user_prompt}\nAssistant:"
    
    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> "Response":
        """
        Async generation via HF Inference Providers
        """
        prompt = self._format_prompt(messages)
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            None,
            lambda: self.client.text_generation(
                prompt=prompt,
                model=self.model_id,
                max_new_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
                temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
                top_p=0.95,
                return_full_text=False,
            ),
        )

        generated_text = self._clean_response(str(generated_text))
        tokens = len(prompt.split()) + len(generated_text.split())
        return Response(content=generated_text, tokens=tokens)
    
    def _clean_response(self, text: str) -> str:
        """Remove special tokens and prefixes"""
        # Remove common end tokens
        for token in ["</s>", "<|endoftext|>", "<|im_end|>", "[/INST]"]:
            text = text.replace(token, "")
        
        # Remove "Assistant:" prefix if present
        text = text.replace("Assistant:", "").strip()
        
        return text
    
    def invoke(self, messages: List[Dict[str, str]], **kwargs):
        """Sync wrapper"""
        import asyncio
        return asyncio.run(self.ainvoke(messages, **kwargs))

class Response:
    """OpenAI-compatible response object"""
    def __init__(self, content: str, tokens: int):
        self.content = content
        self.tokens = tokens

# Dedicated endpoint version (faster, production-grade)
class HFInferenceEndpointLLM:
    """
    Dedicated HF Inference Endpoint
    URL format: https://xxx.us-east-1.aws.endpoints.huggingface.cloud
    """
    
    def __init__(self, endpoint_url: str, api_token: str = None):
        self.api_url = endpoint_url
        self.api_token = api_token or settings.HF_API_TOKEN
        self.client = InferenceClient(
            api_key=self.api_token,
            base_url=self.api_url,
        )
    
    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs):
        prompt = self._format_messages(messages)
        
        loop = asyncio.get_event_loop()
        generated = await loop.run_in_executor(
            None,
            lambda: self.client.text_generation(
                prompt=prompt,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                return_full_text=False,
            ),
        )
        return Response(content=str(generated), tokens=0)
    
    def _format_messages(self, messages):
        # Simple concatenation for dedicated endpoints
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
