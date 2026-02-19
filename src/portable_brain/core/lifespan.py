from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI
from portable_brain.config.app_config import get_service_settings
from portable_brain.common.logging.logger import logger
from portable_brain.common.db.session import create_db_engine_context, parse_db_settings_from_service, DBSettings, DBType
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient, TypedLLMProtocol, LLMProvider
from portable_brain.common.services.embedding_service.text_embedding import TypedTextEmbeddingClient, TextEmbeddingProvider
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient
from portable_brain.common.services.llm_service.llm_client.amazon_nova_client import AsyncAmazonNovaTypedClient
from portable_brain.common.services.embedding_service.text_embedding.gemini_embedding_client import AsyncGenAITextEmbeddingClient
from portable_brain.common.services.droidrun_tools import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
from portable_brain.memory.main_retriever import MemoryRetriever

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
        # NOTE: for now, only Google GenAI and Amazon NOVA clients
        gemini_llm_client = AsyncGenAITypedClient(api_key=settings.GOOGLE_GENAI_API_KEY)
        # wrap around GenAI client for management
        typed_gemini_llm_client = TypedLLMClient(provider=LLMProvider.GOOGLE_GENAI, client=gemini_llm_client)
        app.state.gemini_llm_client = typed_gemini_llm_client
        logger.info(f"LLM client (GOOGLE GENAI) initialized.")

        nova_llm_client = AsyncAmazonNovaTypedClient(api_key=settings.NOVA_API_KEY)
        # wrap around Amazon NOVA client for management
        typed_nova_llm_client = TypedLLMClient(provider=LLMProvider.AMAZON_NOVA, client=nova_llm_client)
        app.state.nova_llm_client = typed_nova_llm_client
        logger.info(f"LLM client (AMAZON NOVA) initialized.")

        # Text embedding models, for now only Google GenAI
        gemini_text_embedding_client = AsyncGenAITextEmbeddingClient(api_key=settings.GOOGLE_GENAI_API_KEY)
        typed_gemini_text_embedding_client = TypedTextEmbeddingClient(provider=TextEmbeddingProvider.GOOGLE_GENAI, client=gemini_text_embedding_client)
        app.state.gemini_text_embedding_client = typed_gemini_text_embedding_client
        logger.info(f"Text embedding client (GOOGLE GENAI) initialized.")

        # DroidRun SDK client (uses same Google GenAI LLM via load_llm)
        droidrun_client = DroidRunClient(api_key=settings.GOOGLE_GENAI_API_KEY)

        # Connect to android device at startup; if fails, exit
        try:
            connected = await droidrun_client.connect()
            if connected:
                app.state.droidrun_client = droidrun_client
                logger.info(f"DroidRun SDK client connected to device {droidrun_client.device_serial}")
            else:
                logger.error("DroidRun SDK client initialized but failed to connect to device. Exiting...")
                raise RuntimeError("DroidRun is a required dependency. Please check.")
        except Exception as e:
            logger.error(f"DroidRun SDK connection failed at startup: {e}. This is a core dependency, please check.")
            raise RuntimeError("DroidRun is a required dependency. Please check.")

        # Observation tracker
        # NOTE: the tracker holds its own instance of dependencies for background tasks
        observation_tracker = ObservationTracker(
            droidrun_client=droidrun_client,
            llm_client=typed_gemini_llm_client,
            text_embedding_client=typed_gemini_text_embedding_client,
            main_db_engine=app.state.main_db_engine
        )
        app.state.observation_tracker = observation_tracker
        logger.info(f"Background observation tracker initialized.")

        # Memory retriever interface
        memory_retriever = MemoryRetriever(
            main_db_engine=app.state.main_db_engine,
            text_embedding_client=typed_gemini_text_embedding_client
        )
        app.state.memory_retriever = memory_retriever
        logger.info(f"Memory retriever initialized.")

        # Execution agent
        execution_agent = ExecutionAgent(
            droidrun_client=droidrun_client,
            gemini_llm_client=gemini_llm_client # NOTE: not the general typed client, only gemini has atool_call() method
        )
        app.state.execution_agent = execution_agent
        logger.info(f"Execution agent initialized.")

        # Retrieval agent
        retrieval_agent = RetrievalAgent(
            memory_retriever=memory_retriever,
            gemini_llm_client=gemini_llm_client # NOTE: not the general typed client, only gemini has atool_call() method
        )
        app.state.retrieval_agent = retrieval_agent
        logger.info(f"Retrieval agent initialized.")

        try:
            # lets FastAPI process requests during yield
            yield
        finally:
            # explicit resource clean up, otherwise automatically cleaned via exit stack
            logger.info("Shutting down service resources...")
            # TODO: add any explicit cleanup / shutdown code here
            # clean up the internal observation tracker
            await observation_tracker.stop_tracking()

        # The AsyncExitStack will automatically call the __aexit__ or registered cleanup
        # methods for all resources entered or pushed to it, in reverse order.
        logger.info("All global resources have been gracefully closed.")
