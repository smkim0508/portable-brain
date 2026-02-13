from google import genai
from google.genai import types
from google.genai.types import ContentEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv

from typing import Type, List, Optional
from pydantic import BaseModel, ValidationError
# use tenacity to retry when desired
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from google import genai # officially recommended import path

from portable_brain.common.services.embedding_service.text_embedding.protocols import PydanticModel, TypedTextEmbeddingProtocol, ProvidesProviderInfo
from portable_brain.common.services.embedding_service.text_embedding.protocols import RateLimitProvider
from portable_brain.common.types.text_embedding_task_types import VALID_GEMINI_TASK_TYPES
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncGenAITextEmbeddingClient(TypedTextEmbeddingProtocol, ProvidesProviderInfo):
    """
    Core Google GenAI Embedding Client.
    """
    def __init__(
        self,
        model_name: str = "gemini-embedding-001", # google genai's default text embedding model
        content_type: str = "RETRIEVAL_DOCUMENT", # choose to differ embedding style, RETRIEVAL_DOCUMENT for text log context search
        embedding_size: int = 1536, # NOTE: use at max 1536 embeddings for now, since pgvector supports upto 2000 dims. Default is 3072, could be explored later.
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
        self.content_type = content_type # content/task type to specialized embeddings
        self.embedding_size = embedding_size
        # Provider metadata for reporting
        self.provider = RateLimitProvider.GOOGLE
        self.model = model_name
        self.retryer = AsyncRetrying(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_fixed(retry_wait),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )

    async def aembed_text(self, text: list[str], task_type: Optional[str] = None) -> list[list[float]]:
        """
        Converts list of text strings into embedding vectors.
        Returns one ContentEmbedding per input text.

        Args:
            text: List of text strings to embed.
            task_type: Optional override for the embedding task type. 
            If None or not a recognized Gemini task type, falls back to the instance's content_type.
        """
        resolved_task_type = task_type if task_type in VALID_GEMINI_TASK_TYPES else self.content_type
        last_exception = None
        attempt_count = 0

        async for attempt in self.retryer:
            attempt_count += 1
            with attempt:
                try:
                    # uses async model
                    result = await self.client.aio.models.embed_content(
                        model=self.model,
                        contents=text, # type: ignore[arg-type] # GenAI SDK accepts list[str] at runtime
                        # sets configs: task type and embedding size
                        config=types.EmbedContentConfig(task_type=resolved_task_type, output_dimensionality=self.embedding_size),
                    )
                    if result:
                        embeddings: List[ContentEmbedding] | None = result.embeddings
                        if embeddings:
                            # return list of embedding vectors (float), filtering out None embeddings
                            return [
                                embedding.values
                                for embedding in embeddings
                                if embedding is not None and embedding.values is not None
                            ]
                    raise ValueError("No embeddings returned from API")
                except Exception as e:
                    last_exception = e
                    # Let tenacity handle retry/terminal re-raise
                    raise

        raise RuntimeError(
            f"aembed_text() reached unexpected fallthrough after {attempt_count} attempts; "
            f"retryer likely yielded no final exception and no success. last_exc={type(last_exception).__name__ if last_exception else None}"
        )
