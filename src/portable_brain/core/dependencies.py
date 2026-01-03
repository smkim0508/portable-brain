from typing import Any, Optional
from portable_brain.common.logging.logger import logger
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncEngine
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient, TypedLLMProtocol
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient
from portable_brain.common.services.droidrun_tools import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker

# This is the location to conveniently return any app lifetime dependencies to be used in routes
# TODO: add more dependencies as needed
def get_main_db_engine(request: Request) -> AsyncEngine:
    """
    FastAPI dependency to get the shared main DB engine from the application state.
    """
    return request.app.state.main_db_engine

def get_llm_client(request: Request) -> TypedLLMProtocol:
    """
    FastAPI dependency to get the shared LLM client from the application state.
    """
    return request.app.state.llm_client

def get_droidrun_client(request: Request) -> DroidRunClient:
    """
    FastAPI dependency to get the shared DroidRun SDK client from the application state.
    """
    return request.app.state.droidrun_client

def get_observation_tracker(request: Request) -> ObservationTracker:
    """
    FastAPI dependency to get the shared observation tracker from the application state.
    """
    return request.app.state.observation_tracker
