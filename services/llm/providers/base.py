from abc import ABC, abstractmethod
from typing import Dict, Any

from schemas import GenerateRequest, GenerateResponse, HealthResponse

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        pass