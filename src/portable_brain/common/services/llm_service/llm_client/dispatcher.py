# dispatcher for LLM clients, currently only Google Gemini AI, but scalable to other LLM providers
# NOTE: this generic LLM wrapper is not necessary right now since we only have Google Gemini for service.

from enum import Enum, auto
from typing import Type, TypeVar
from pydantic import BaseModel
from .protocols import TypedLLMProtocol

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

# NOTE: to be expanded with more services if desired
class LLMProvider(Enum):
    GOOGLE_GENAI = auto() # just need a unique identifier
    AMAZON_NOVA = auto()

class TypedLLMClient:
    def __init__(self, provider: LLMProvider, client: TypedLLMProtocol):
        self.provider = provider
        self.client = client

    async def acreate(
        self,
        response_model: Type[PydanticModel],
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> PydanticModel:
        return await self.client.acreate(
            response_model=response_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs
        )
