# test route for db-related functionality
# NOTE: currently supports all dbs, both vector db and structured db; may be split in the future.

import time
from uuid import uuid4
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
from portable_brain.api.request_models.tests import TestRequest, TestEmbeddingRequest, SimilarEmbeddingRequest, SaveObservationRequest
# crud
from portable_brain.common.db.crud.memory.text_embeddings_crud import find_similar_embeddings
from portable_brain.common.db.crud.memory.structured_memory_crud import save_observation_to_structured_memory
# observation DTOs
from portable_brain.monitoring.background_tasks.types.observation.observations import ShortTermPreferencesObservation
from datetime import datetime

router = APIRouter(prefix="/db-test", tags=["Tests"])

@router.post("/save-observation")
async def save_observation(
    request: SaveObservationRequest,
    main_db_engine: AsyncEngine = Depends(get_main_db_engine)
):
    # mock a ShortTermPreferencesObservation with the provided observation node
    mock_observation = ShortTermPreferencesObservation(
        id=str(uuid4()),
        importance=0.5,
        created_at=datetime.now(),
        source_id="com.instagram.android",
        edge="prefers",
        node=request.observation_node,
        recurrence=1,
    )
    try:
        await save_observation_to_structured_memory(
            observation=mock_observation,
            main_db_engine=main_db_engine
        )
        return {"message": f"Successfully saved observation '{mock_observation.id}' to structured memory"}
    except Exception as e:
        logger.error(f"Error saving observation to structured memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving observation to structured memory: {e}")
