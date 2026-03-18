"""
Local LLM Generation with HuggingFace
Uses quantized models for reasonable GPU/CPU requirements
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Tuple
import torch
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class HuggingFaceLLM:
    """
    Local LLM for RAG generation
    Recommended: "HuggingFaceH4/zephyr-7b-beta" or "mistralai/Mistral-7B-Instruct-v0.2"
    For CPU: Use "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (slower but works)
    """
    
    def __init__(self, model_name: str = None):
        selected_name = model_name or settings.LLM_MODEL
        shorthand = {
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
            "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
            "gemma-2b": "google/gemma-2b-it",
            "phi-2": "microsoft/phi-2",
        }
        self.model_name = shorthand.get(selected_name, selected_name)
        
        logger.info(f"Loading LLM: {self.model_name}")
        
        # Quantization config (load 4-bit to save VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if torch.cuda.is_available() else None
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        logger.info("LLM loaded successfully")
    
    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format for instruction-tuned models
        Different models need different formats!
        """
        # Mistral/Zephyr format
        if "mistral" in self.model_name.lower() or "zephyr" in self.model_name.lower():
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        # Llama-2 format
        elif "llama-2" in self.model_name.lower():
            return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        
        # Generic chat format
        else:
            return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """
        Generate response from messages
        Returns: (generated_text, token_count)
        """
        # Extract system and user messages
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        user_prompt = "\n".join(user_msgs)
        
        # Format prompt
        prompt = self._format_prompt(system_msg, user_prompt)
        
        # Generate
        outputs = self.pipe(
            prompt,
            return_full_text=False,
            clean_up_tokenization_spaces=True
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Count tokens (approximate)
        tokens = len(self.tokenizer.encode(prompt + generated_text))
        
        return generated_text.strip(), tokens
    
    async def ainvoke(self, messages: List[Dict[str, str]]):
        """Async wrapper"""
        import asyncio
        loop = asyncio.get_event_loop()
        content, tokens = await loop.run_in_executor(None, self.generate, messages)
        
        # Return OpenAI-compatible response object
        class Response:
            def __init__(self, content, tokens):
                self.content = content
                self.tokens = tokens
        
        return Response(content, tokens)

# Model recommendations
LLM_MODELS = {
    "quality_gpu": "mistralai/Mistral-7B-Instruct-v0.2",      # Needs 16GB VRAM or 4-bit quant
    "balanced_gpu": "HuggingFaceH4/zephyr-7b-beta",           # Similar, different training
    "cpu_friendly": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     # Works on CPU, lower quality
    "fast_gpu": "google/gemma-2b-it",                           # 2B params, decent quality
}
