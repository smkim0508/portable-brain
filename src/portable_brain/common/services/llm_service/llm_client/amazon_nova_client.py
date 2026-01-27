# Amazon NOVA Client
import os
from openai import OpenAI # Nova Model uses OpenAI's API Schema
from typing import Type
from pydantic import BaseModel, ValidationError
# use tenacity to retry when desired
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from .protocols import PydanticModel, TypedLLMProtocol, ProvidesProviderInfo
from .protocols import RateLimitProvider
import asyncio
from concurrent.futures import ThreadPoolExecutor

# NOTE: this uses the public Gemini API with an API key, not Vertex AI.
# Set up this client with API key during app initialization
# TODO: "strict" JSON/Pydantic output is only supported for Vertex AI clients; set up manual validation to catch malformed JSON outputs before crashing Pydantic validation, or loosen validation.
class AsyncAmazonNovaTypedClient(TypedLLMProtocol, ProvidesProviderInfo):
    def __init__(
        self,
        model_name: str = "nova-2-lite-v1", # given from documentation, could be swapped depending on rate limits / pricing
        *,
        api_key: str | None = None,
        retry_attempts: int = 2,
        retry_wait: float = 0.1,
        retry_on: type[Exception] = ValidationError, # only retry when LLM fails to meet Pydantic validation
    ):
        # Create shared client in __init__ for FastAPI (ASGI)
        # FastAPI runs in a single event loop, so sharing the client is safe and efficient
        # This enables connection pooling and reduces overhead compared to creating a new client per request
        self.client = OpenAI(api_key=api_key, base_url="https://api.nova.amazon.com/v1") # Nova-compatible base url
        self.model_name = model_name
        # Provider metadata for reporting
        self.provider = RateLimitProvider.AWS
        self.model = model_name
        self.retryer = AsyncRetrying(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_fixed(retry_wait),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )

    async def acreate(
        self,
        response_model: Type[PydanticModel],
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> PydanticModel:
        prompt = f"{system_prompt}\n\n{user_prompt}"

        last_exception = None
        attempt_count = 0

        async for attempt in self.retryer:
            attempt_count += 1
            with attempt: # let tenacity see context of each attempt instead of swallowing until the last
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"}, # enforces JSON mode
                        **kwargs,
                    )
                    parsed = getattr(resp, "parsed", None)
                    if parsed is not None:
                        return parsed  # type: ignore[return-value]
                    
                    text = getattr(resp, "text", None)
                    if isinstance(text, str) and text.strip():
                        return response_model.model_validate_json(text)

                    # Force a retryable failure when output is empty/invalid
                    raise ValueError("LLM response was empty or not parseable to JSON.")
                
                except ValidationError as e:
                    last_exception = e
                    # Let tenacity handle retry/terminal re-raise
                    raise
                except Exception as e:
                    last_exception = e
                    # Let tenacity handle retry/terminal re-raise
                    raise
        
        # This should never be reached, but if it is, provide better error info
        # NOTE: **IMPORTANT** If we get here, the loop ran zero times (misconfigured retryer) or exited cleanly without return.
        raise RuntimeError(
            f"acreate() reached unexpected fallthrough after {attempt_count} attempts; "
            f"retryer likely yielded no final exception and no success. last_exc={type(last_exception).__name__ if last_exception else None}"
        )
