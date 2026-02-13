# test route to fetch data from droidrun client or observation tracker

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

router = APIRouter(prefix="/tracker-test", tags=["Tests"])

@router.get("/get-raw-tree")
async def get_raw_accessibility_tree(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client)
):
    """
    Get the raw accessibility tree from the current screen.
    Returns all available information from the a11y tree including:
    - raw_tree: Complete unfiltered accessibility tree
    - formatted_text: Human-readable indexed UI description
    - focused_element: Currently focused element
    - ui_elements: Parsed list of UI elements
    - phone_state: Current app package, activity, and editable status
    """
    try:
        logger.info("Fetching raw accessibility tree from current screen")
        state = await droidrun_client.get_current_state()
        raw_tree = await droidrun_client.get_raw_tree()

        return {
            "message": "Successfully retrieved raw accessibility tree",
            # "raw_tree": state["raw_tree"],
            "formatted_text": state["formatted_text"],
            "focused_element": state["focused_element"],
            "ui_elements": state["ui_elements"],
            "phone_state": state["phone_state"],
            "timestamp": state["timestamp"]
        }
        # return {
        #     "message": "ok",
        #     "raw_tree": raw_tree["raw_tree"]
        # }
    except Exception as e:
        logger.error(f"Error fetching raw accessibility tree: {e}")
        return {"message": f"Error fetching raw accessibility tree: {e}"}, 500


@router.get("/get-droidrun-state")
async def get_droidrun_state(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client)
):
    """
    Gets the raw UI state given by DroidRun SDK.
    """
    try:
        logger.info("Fetching raw UI state from current screen")
        state = await droidrun_client.get_raw_state()
        raw_tree = await droidrun_client.get_raw_tree()

        return {
            "message": "Successfully retrieved raw accessibility tree",
            "raw_state": state
        }
    
    except Exception as e:
        logger.error(f"Error fetching raw UI state: {e}")
        return {"message": f"Error fetching raw UI state from DroidRun: {e}"}, 500
