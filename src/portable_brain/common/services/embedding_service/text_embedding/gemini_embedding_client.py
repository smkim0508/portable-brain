from google import genai
from google.genai import types
from google.genai.types import ContentEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv

from typing import Type, List
from pydantic import BaseModel, ValidationError
# use tenacity to retry when desired
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from google import genai # officially recommended import path

from portable_brain.common.services.embedding_service.text_embedding.protocols import PydanticModel, TypedTextEmbeddingProtocol, ProvidesProviderInfo
from portable_brain.common.services.embedding_service.text_embedding.protocols import RateLimitProvider
import asyncio
from concurrent.futures import ThreadPoolExecutor

class GoogleGenAIEmbeddingClient(TypedTextEmbeddingProtocol, ProvidesProviderInfo):
    """
    Core Google GenAI Embedding Client.
    """
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite", # given from documentation, could be swapped depending on rate limits / pricing
        content_type: str = "RETRIEVAL_DOCUMENT", # choose to differ embedding style
        *,
        api_key: str | None = None,
        retry_attempts: int = 2,
        retry_wait: float = 0.1,
        retry_on: type[Exception] = ValidationError, # only retry when LLM fails to meet Pydantic validation
    ):
        # Create shared client in __init__ for FastAPI (ASGI)
        # FastAPI runs in a single event loop, so sharing the client is safe and efficient
        # This enables connection pooling and reduces overhead compared to creating a new client per request
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.content_type = content_type
        # Provider metadata for reporting
        self.provider = RateLimitProvider.GOOGLE
        self.model = model_name
        self.retryer = AsyncRetrying(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_fixed(retry_wait),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )

    async def aembed_text(self, text: list[str]) -> list[ContentEmbedding]:
        """
        Converts list of text strings into embedding vectors.
        Returns one ContentEmbedding per input text.
        """
        last_exception = None
        attempt_count = 0

        async for attempt in self.retryer:
            attempt_count += 1
            with attempt:
                try:
                    # NOTE: not asynchronous yet
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=text, # type: ignore[arg-type] # GenAI SDK accepts list[str] at runtime
                        config=types.EmbedContentConfig(task_type=self.content_type),
                    )
                    if result and result.embeddings:
                        return result.embeddings
                    raise ValueError("No embeddings returned from API")
                except Exception as e:
                    last_exception = e
                    # Let tenacity handle retry/terminal re-raise
                    raise

        raise RuntimeError(
            f"aembed_text() reached unexpected fallthrough after {attempt_count} attempts; "
            f"retryer likely yielded no final exception and no success. last_exc={type(last_exception).__name__ if last_exception else None}"
        )
