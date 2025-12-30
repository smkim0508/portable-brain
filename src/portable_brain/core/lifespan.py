from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI
from portable_brain.config.app_config import get_service_settings
from portable_brain.common.logging.logger import logger
from portable_brain.common.db.session import create_db_engine_context, parse_db_settings_from_service, DBSettings, DBType
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient, TypedLLMProtocol, LLMProvider
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient
from portable_brain.common.services.droidrun_tools import DroidRunClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the service's startup and shutdown events.
    Uses the AsyncExitStack to clean up resources.
    Register resources to the app state to be used as dependencies.

    NOTE:
    - Use stack.enter_async_context when the resource has __aenter__ and __aexit__ support
    - Use stack.push_async_context to register the clean up method only
    """
    # default start up message
    logger.info(f"Starting Portable-Brain service!")

    # initialize resources during start up
    logger.info("Initializing service resources...")
    settings = get_service_settings()

    async with AsyncExitStack() as stack:
        
        # TODO: add any initialization code here

        # Main db engine
        # parse main db settings, and create engine
        main_db_settings = parse_db_settings_from_service(settings, DBType.MainDB)
        app.state.main_db_engine = await stack.enter_async_context(
            create_db_engine_context(
                db_settings=main_db_settings
            )
        )
        logger.info("Main database engine initialized.")

        # LLM clients (no need for resource clean up)

        # NOTE: for now, only Google GenAI client
        google_llm_client = AsyncGenAITypedClient(api_key=settings.GOOGLE_GENAI_API_KEY)
        # wrap around GenAI client for management
        typed_llm_client = TypedLLMClient(provider=LLMProvider.GOOGLE_GENAI, client=google_llm_client)
        app.state.llm_client = typed_llm_client
        logger.info(f"LLM client (GOOGLE GENAI) initialized.")

        # DroidRun SDK client (no auth required)
        droidrun_client = DroidRunClient()
        app.state.droidrun_client = droidrun_client
        logger.info("DroidRun SDK client initialized.")

        try:
            # lets FastAPI process requests during yield
            yield
        finally:
            # explicit resource clean up, otherwise automatically cleaned via exit stack
            logger.info("Shutting down service resources...")
            # TODO: add any explicit cleanup / shutdown code here

        # The AsyncExitStack will automatically call the __aexit__ or registered cleanup
        # methods for all resources entered or pushed to it, in reverse order.
        logger.info("All global resources have been gracefully closed.")
