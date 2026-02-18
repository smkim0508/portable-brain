from typing import Any, Optional
from portable_brain.common.logging.logger import logger
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncEngine
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient, TypedLLMProtocol
from portable_brain.common.services.embedding_service.text_embedding import TypedTextEmbeddingClient
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient
from portable_brain.common.services.droidrun_tools import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent

# This is the location to conveniently return any app lifetime dependencies to be used in routes
# TODO: add more dependencies as needed
def get_main_db_engine(request: Request) -> AsyncEngine:
    """
    FastAPI dependency to get the shared main DB engine from the application state.
    """
    return request.app.state.main_db_engine

def get_gemini_llm_client(request: Request) -> TypedLLMClient:
    """
    FastAPI dependency to get the shared Google GenAI LLM client from the application state.
    """
    return request.app.state.gemini_llm_client

def get_nova_llm_client(request: Request) -> TypedLLMClient:
    """
    FastAPI dependency to get the shared Amazon Nova LLM client from the application state.
    """
    return request.app.state.nova_llm_client

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

def get_gemini_text_embedding_client(request: Request) -> TypedTextEmbeddingClient:
    """
    FastAPI dependency to get the shared Google Gen AI text embedding client from the application state.
    """
    return request.app.state.gemini_text_embedding_client

def get_execution_agent(request: Request) -> ExecutionAgent:
    """
    FastAPI dependency to get the shared execution agent from the application state.
    """
    return request.app.state.execution_agent
