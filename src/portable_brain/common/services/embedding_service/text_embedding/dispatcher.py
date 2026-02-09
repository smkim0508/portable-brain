# dispatcher for text embedding clients, currently only Google Gemini AI, but scalable to other LLM providers
# NOTE: this generic LLM wrapper is not necessary right now since we only have Google Gemini for service.

from enum import Enum, auto
from typing import Type, TypeVar
from pydantic import BaseModel
from portable_brain.common.services.embedding_service.text_embedding.protocols import TypedTextEmbeddingProtocol

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

# NOTE: to be expanded with more services if desired
class TextEmbeddingProvider(Enum):
    GOOGLE_GENAI = auto() # just need a unique identifier

class TypedTextEmbeddingClient:
    def __init__(self, provider: TextEmbeddingProvider, client: TypedTextEmbeddingProtocol):
        self.provider = provider
        self.client = client

    async def aembed_text(
        self,
        text: list[str],
        **kwargs
    ) -> list[list[float]]:
        return await self.client.aembed_text(
            text=text,
            **kwargs
        )
