# test route for llm-related functionalities

import time
from fastapi import APIRouter, Depends, Response, status, HTTPException, Query
from portable_brain.common.logging.logger import logger
from portable_brain.common.db.session import get_async_session_maker
# dependencies
from portable_brain.core.dependencies import (
    get_main_db_engine,
    get_droidrun_client,
    get_gemini_llm_client,
    get_nova_llm_client,
    get_observation_tracker,
    get_gemini_text_embedding_client
)
# services and clients
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient
from portable_brain.common.services.embedding_service.text_embedding import TypedTextEmbeddingClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# response models
from portable_brain.api.response_models.tests import TestResponse, SimilarEmbeddingResponse
# request body models
from portable_brain.api.request_models.tests import TestRequest, TestEmbeddingRequest, SimilarEmbeddingRequest
# crud
from portable_brain.common.db.crud.memory.text_embeddings_crud import find_similar_embeddings

router = APIRouter(prefix="/llm-test", tags=["Tests"])

# NOTE: to be populated
