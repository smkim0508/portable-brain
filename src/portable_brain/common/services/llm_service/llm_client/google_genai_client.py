# The core async set up for Google's GenAI LLM client
# NOTE: Can be swapped for different LLM providers if necessary

from typing import Type, Any, Callable, Awaitable
from pydantic import BaseModel, ValidationError
# use tenacity to retry when desired
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from google import genai # officially recommended import path
from google.genai import types

from .protocols import PydanticModel, TypedLLMProtocol, ProvidesProviderInfo
from .protocols import RateLimitProvider
import asyncio
from concurrent.futures import ThreadPoolExecutor

# NOTE: this uses the public Gemini API with an API key, not Vertex AI.
# Set up this client with API key during app initialization
# TODO: "strict" JSON/Pydantic output is only supported for Vertex AI clients; set up manual validation to catch malformed JSON outputs before crashing Pydantic validation, or loosen validation.
class AsyncGenAITypedClient(TypedLLMProtocol, ProvidesProviderInfo):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite", # given from documentation, could be swapped depending on rate limits / pricing
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
        # Provider metadata for reporting
        self.provider = RateLimitProvider.GOOGLE
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
        """
        Helper to create an async response from the LLM and parse it into a structured Pydantic model output.
        Supports retries to ensure LLM meets Pydantic validation.
        """

        prompt = f"{system_prompt}\n\n{user_prompt}"

        last_exception = None
        attempt_count = 0

        async for attempt in self.retryer:
            attempt_count += 1
            with attempt: # let tenacity see context of each attempt instead of swallowing until the last
                try:
                    resp = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=prompt, # auto-wrapped in a content object
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": response_model,
                        },
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
    
    async def atool_call(
        self,
        system_prompt: str,
        user_prompt: str,
        function_declarations: list[dict],
        tool_executors: dict[str, Callable[..., Awaitable[Any]]],
        max_turns: int = 5,
        **kwargs,
    ) -> str:
        """
        Helper to create async requests to LLM for tool calling purposes.
        Supports multi-turn execution and retries on API failures.

        Returns: the final text response from the LLM after all tool calls are resolved.
        NOTE: no tenacity retryer yet; to be implemented as an inner loop wrapper.
        """
        # wrap function declarations in tool and config objects
        tools = types.Tool(function_declarations=function_declarations) # type: ignore
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[tools],
        )

        # define user prompt, wrapped in Content to track multi-turn conversation
        contents: list[types.Content] = [
            types.Content(
                role="user", parts=[types.Part(text=user_prompt)]
            )
        ]

        for _turn in range(max_turns):
            # call LLM with current conversation history
            resp = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents, # type: ignore
                config=config,
            )

            # extract the response content
            response_content = resp.candidates[0].content # type: ignore

            # check if the LLM wants to call a tool
            tool_call = getattr(response_content.parts[0], "function_call", None) if response_content and response_content.parts else None

            if tool_call is None or tool_call.name is None:
                # NOTE: LLM responded with text, not a tool call, so we're done
                return resp.text or ""

            # LLM requested a tool call, dispatch to the appropriate executor
            tool_name = tool_call.name
            tool_args = dict(tool_call.args) if tool_call.args else {}

            if tool_name not in tool_executors:
                raise ValueError(f"LLM requested unknown tool '{tool_name}'. Available: {list(tool_executors.keys())}")

            # execute the tool
            try:
                result = await tool_executors[tool_name](**tool_args)
                tool_response = {"result": result}
            except Exception as e:
                # send the error back to the LLM so it can recover or explain
                tool_response = {"error": str(e)}

            # build the function response part
            function_response_part = types.Part.from_function_response(
                name=tool_name,
                response=tool_response,
            )

            # append model's tool call + our execution result to conversation history
            contents.append(response_content) # type: ignore
            contents.append(types.Content(role="user", parts=[function_response_part]))

        # exhausted max turns without getting a text response
        raise RuntimeError(
            f"atool_call() exhausted {max_turns} turns without receiving a final text response from LLM."
        )
