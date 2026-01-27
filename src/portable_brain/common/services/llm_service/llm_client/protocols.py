# protocals for LLM clients

from typing import Protocol, TypeVar, Type, runtime_checkable
from pydantic import BaseModel
from enum import Enum

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

# Ensures that all LLM clients implement this protocol
class TypedLLMProtocol(Protocol):
    async def acreate(
        self,
        response_model: Type[PydanticModel],
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> PydanticModel: ...

class RateLimitProvider(str, Enum):
    """Enumeration of supported rate limit providers."""
    GOOGLE = "google"
    AWS = "aws"

@runtime_checkable
class ProvidesProviderInfo(Protocol):
    """Optional protocol for exposing provider/model metadata for reporting."""
    provider: RateLimitProvider
    model: str
