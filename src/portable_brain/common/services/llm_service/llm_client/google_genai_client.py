# The core async set up for Google's GenAI LLM client
# NOTE: Can be swapped for different LLM providers if necessary

from typing import Type, Any, Callable, Awaitable, Optional
from pydantic import BaseModel, ValidationError
# use tenacity to retry when desired
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from google import genai # officially recommended import path
from google.genai import types

from .protocols import PydanticModel, TypedLLMProtocol, ProvidesProviderInfo
from .protocols import RateLimitProvider
import asyncio
from concurrent.futures import ThreadPoolExecutor

# logger to debug tool calling
from portable_brain.common.logging.logger import logger

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
    
    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # remove opening fence (```json or ```)
            first_newline = stripped.index("\n") if "\n" in stripped else len(stripped)
            stripped = stripped[first_newline + 1:]
            # remove closing fence
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
        return stripped

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert tool results into JSON-serializable primitives."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        return str(obj)

    async def atool_call(
        self,
        system_prompt: str,
        user_prompt: str,
        function_declarations: list[dict],
        tool_executors: dict[str, Callable[..., Awaitable[Any]]],
        response_model: Optional[Type[PydanticModel]] = None,
        max_turns: int = 5,
        **kwargs,
    ) -> str | PydanticModel:
        """
        Helper to create async requests to LLM for tool calling purposes.
        Supports multi-turn execution and retries on API failures.

        Returns: the final text response from the LLM after all tool calls are resolved.
        - Optionally, the final response is parsed into a Pydantic model if a response_model is provided.
        NOTE: no tenacity retryer yet; to be implemented as an inner loop wrapper.
        """
        # wrap function declarations in tool and config objects
        tools = types.Tool(function_declarations=function_declarations) # type: ignore

        # NOTE: it is possible to optionally add pydantic schema here, but this might cause competitinng output goals for LLM.
        # disallowed until future experiments
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
                text = resp.text or ""
                # if response model is provided, try to validate and return
                if response_model:
                    # NOTE: need to add retry logic if this fails in production
                    try:
                        cleaned = self._strip_markdown_fences(text)
                        return response_model.model_validate_json(cleaned)
                    except Exception as validation_error:
                        logger.warning(f"Failed to validate LLM response as {response_model.__name__}: {validation_error}\nRaw LLM output: {text[:500]}")
                return text

            # LLM requested a tool call, dispatch to the appropriate executor
            tool_name = tool_call.name
            tool_args = dict(tool_call.args) if tool_call.args else {}
            logger.info(f"[atool_call] Turn {_turn + 1}: LLM called '{tool_name}' with args: {tool_args}")

            if tool_name not in tool_executors:
                raise ValueError(f"LLM requested unknown tool '{tool_name}'. Available: {list(tool_executors.keys())}")

            # execute the tool
            try:
                result = await tool_executors[tool_name](**tool_args)
                tool_response = {"result": self._make_serializable(result)}
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
