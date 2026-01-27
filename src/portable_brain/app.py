from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional, Dict
from portable_brain.common.logging.logger import logger
from portable_brain.config.app_config import get_service_settings
from portable_brain.core.lifespan import lifespan
from fastapi.middleware.cors import CORSMiddleware
from portable_brain.common.db.session import get_async_session_maker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from portable_brain.api.routes.test_route import router as test_router
from portable_brain.api.routes.monitoring_background_tasks import router as monitoring_router
from portable_brain.common.services.llm_service.llm_client import TypedLLMProtocol
from portable_brain.agent_service.common.types.test_llm_outputs import TestLLMOutput
from portable_brain.middleware.logging_middleware import LoggingMiddleware
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker

from portable_brain.core.dependencies import (
    get_gemini_llm_client,
    get_nova_llm_client,
    get_main_db_engine,
    get_droidrun_client,
    get_observation_tracker
)

# disable FastAPI docs for production/deployment
is_local = get_service_settings().INCLUDE_DOCS
logger.info(f"is_local (include FastAPI docs?): {is_local}")

docs_config: dict[str, Any] = {
    "docs_url": "/docs" if is_local else None,
    "redoc_url": "/redoc" if is_local else None,
    "openapi_url": "/openapi.json" if is_local else None,
}

# main app, asgi entrypoint
app = FastAPI(
    title="Portable Brain Service",
    description="Experimental service for testing different memroy structures to observe HCI data",
    version="0.1.0",
    lifespan=lifespan,
    **docs_config,  
)

# add CORS middleware
# TODO: make this more restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# add logging middleware for requests
app.add_middleware(LoggingMiddleware)

# add routes
app.include_router(test_router)
app.include_router(monitoring_router)

# test endpoint
@app.get("/", tags=["Application"])
async def root():
    return {"message": "Hello World"}

# health endpoint
@app.get("/health", tags=["Application"])
async def health(
    main_db_engine: AsyncEngine = Depends(get_main_db_engine),
    gemini_llm_client: TypedLLMProtocol = Depends(get_gemini_llm_client),
    nova_llm_client: TypedLLMProtocol = Depends(get_nova_llm_client),
    droidrun_client: DroidRunClient = Depends(get_droidrun_client)
):
    """
    Comprehensive health check for all services.
    Checks all services independently and returns detailed status for each.
    LLM checks are disabled by default in production to avoid API costs.
    """

    health_status = {
        "status": "healthy",
        "services": {}
    }

    # Check database connection (always enabled)
    db_healthy = False
    main_session_maker = get_async_session_maker(main_db_engine)
    try:
        async with main_session_maker() as session:
            await session.execute(text("SELECT 1"))
        db_healthy = True
        health_status["services"]["database"] = {
            "status": "healthy",
            "message": "Connected to main database"
        }
        logger.info("Database health check passed")
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "message": f"Unable to connect: {str(e)}"
        }
        logger.error(f"Database health check failed: {e}")

    # Check LLM connections (only if enabled via config)
    gemini_healthy = True  # Default to true if check is disabled
    nova_healthy = True    # Default to true if check is disabled
    allow_llm_check = get_service_settings().HEALTH_CHECK_LLM

    if allow_llm_check:
        # Check Google Gemini
        gemini_healthy = False
        try:
            gemini_response = await gemini_llm_client.acreate(
                response_model=TestLLMOutput,
                system_prompt="Are you connected?",
                user_prompt="Respond with 'True'.",
            )
            gemini_healthy = True
            health_status["services"]["gemini_llm"] = {
                "status": "healthy",
                "message": "Connected to Google Gemini LLM"
            }
            logger.info(f"Gemini LLM health check passed, response: {gemini_response}")
        except Exception as e:
            health_status["services"]["gemini_llm"] = {
                "status": "unhealthy",
                "message": f"Unable to connect: {str(e)}"
            }
            logger.error(f"Gemini LLM health check failed: {e}")

        # Check Amazon Nova
        nova_healthy = False
        try:
            nova_response = await nova_llm_client.acreate(
                response_model=TestLLMOutput,
                system_prompt="Are you connected?",
                user_prompt="Respond with 'True'.",
            )
            nova_healthy = True
            health_status["services"]["nova_llm"] = {
                "status": "healthy",
                "message": "Connected to Amazon Nova LLM"
            }
            logger.info(f"Nova LLM health check passed, response: {nova_response}")
        except Exception as e:
            health_status["services"]["nova_llm"] = {
                "status": "unhealthy",
                "message": f"Unable to connect: {str(e)}"
            }
            logger.error(f"Nova LLM health check failed: {e}")
    else:
        health_status["services"]["gemini_llm"] = {
            "status": "skipped",
            "message": "LLM health check disabled in production."
        }
        health_status["services"]["nova_llm"] = {
            "status": "skipped",
            "message": "LLM health check disabled in production."
        }

    # Check DroidRun connection
    droidrun_healthy = False
    try:
        # Check if connected (should already have been established at startup)
        if not droidrun_client._connected:
            health_status["services"]["droidrun"] = {
                "status": "unhealthy",
                "message": "Not connected to device (failed at startup)",
                "device_serial": droidrun_client.device_serial
            }
        else:
            # Simple ping to Portal to verify connection is still alive
            ping_result = await droidrun_client.tools.ping()

            # Get current device state for additional validation
            state = await droidrun_client.get_current_state()
            current_app = state['phone_state'].get('packageName', 'Unknown')

            droidrun_healthy = True
            health_status["services"]["droidrun"] = {
                "status": "healthy",
                "message": f"Connected to device {droidrun_client.device_serial}",
                "current_app": current_app,
                "portal_version": ping_result.get("version", "unknown"),
                "DroidAgent": "healthy" if droidrun_client.llm and not droidrun_client.disable_llm else "disabled"
            }
            logger.info(f"DroidRun health check passed, current app: {current_app}")

    except Exception as e:
        health_status["services"]["droidrun"] = {
            "status": "unhealthy",
            "message": f"Connection lost or error: {str(e)}",
            "device_serial": droidrun_client.device_serial
        }
        logger.error(f"DroidRun health check failed: {e}")

    # Set overall status based on all service checks
    if not (db_healthy and gemini_healthy and nova_healthy and droidrun_healthy):
        health_status["status"] = "unhealthy"

    return health_status
