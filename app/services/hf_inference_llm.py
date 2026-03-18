"""
HuggingFace Inference Providers client for text generation.
"""

import asyncio
from typing import List, Dict
import logging
import httpx
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
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    }

    def __init__(self, model_id: str = None, api_token: str = None):
        self.model_id = model_id or settings.LLM_MODEL
        self.api_token = api_token or settings.HF_API_TOKEN

        # Resolve shorthand to full model ID
        if self.model_id in self.FREE_MODELS:
            self.model_id = self.FREE_MODELS[self.model_id]

        self.client = InferenceClient(api_key=self.api_token)

        logger.info(f"HF Inference LLM: {self.model_id}")

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for specific model
        Each model has different chat template!
        """
        # Extract system and user
        system = next((m["content"]
                      for m in messages if m["role"] == "system"), "")
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
        Async generation via HF Inference Providers.
        Prefer conversational API when available, fallback to text-generation.
        """
        loop = asyncio.get_event_loop()

        def _chat_completion_call():
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=kwargs.get(
                    "temperature", settings.LLM_TEMPERATURE),
                max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", None)
            if tokens is None:
                tokens = len(content.split())
            return Response(content=content.strip(), tokens=tokens)

        def _text_generation_call():
            prompt = self._format_prompt(messages)
            generated = self.client.text_generation(
                prompt=prompt,
                model=self.model_id,
                max_new_tokens=kwargs.get(
                    "max_tokens", settings.LLM_MAX_TOKENS),
                temperature=kwargs.get(
                    "temperature", settings.LLM_TEMPERATURE),
                top_p=0.95,
                return_full_text=False,
            )
            generated_text = self._clean_response(str(generated))
            tokens = len(prompt.split()) + len(generated_text.split())
            return Response(content=generated_text, tokens=tokens)

        try:
            return await loop.run_in_executor(None, _chat_completion_call)
        except Exception as chat_err:
            logger.warning(
                "Chat completion failed for %s, falling back to text-generation: %s", self.model_id, chat_err)
            return await loop.run_in_executor(None, _text_generation_call)

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
        self.model_id = settings.LLM_MODEL
        self.is_openai_chat_endpoint = self.api_url.rstrip(
            "/").endswith("/chat/completions")

        if not self.api_token:
            if "featherless" in self.api_url.lower():
                raise ValueError(
                    "Missing Featherless credentials. Set FEATHERLESS_API_KEY for Featherless chat completions."
                )
            raise ValueError(
                "Missing API token for LLM endpoint. Set HF_API_TOKEN."
            )

        if self.is_openai_chat_endpoint:
            self.client = None
            return

        self.client = InferenceClient(
            api_key=self.api_token,
            base_url=self.api_url,
        )

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs):
        if self.is_openai_chat_endpoint:
            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
                "max_tokens": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            }
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)

            if response.status_code == 401:
                raise ValueError(
                    "Featherless rejected the credentials (401). Verify FEATHERLESS_API_KEY and provider account access."
                )

            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = len(content.split())

            return Response(content=content, tokens=total_tokens)

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
