# protocals for LLM clients

from typing import Protocol, TypeVar, Type, runtime_checkable
from pydantic import BaseModel
from enum import Enum

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

# Ensures that all text embedding clients implement this protocol
class TypedTextEmbeddingProtocol(Protocol):
    async def aembed_text(
        self,
        text: str,
        **kwargs
    ) -> PydanticModel: ...

# NOTE: this rate limit provider is not currently in use, for future purposes.
class RateLimitProvider(str, Enum):
    """Enumeration of supported rate limit providers."""
    GOOGLE = "google"

@runtime_checkable
class ProvidesProviderInfo(Protocol):
    """Optional protocol for exposing provider/model metadata for reporting."""
    provider: RateLimitProvider
    model: str
