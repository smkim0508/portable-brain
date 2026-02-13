# test route for basic app functions like router + dependency injection

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

router = APIRouter(prefix="/general-test", tags=["Tests"])

@router.get("/first-test")
async def first_test():
    logger.info("first test route")
    return {"message": "this is the first test route"}

@router.get("/second-test")
async def second_test(main_db_engine: AsyncEngine = Depends(get_main_db_engine)):
    logger.info("second test route, trying to inject db session")
    main_session_maker = get_async_session_maker(main_db_engine)

    try:
        async with main_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        logger.info(f"error: {e}")
        return {"message": "unable to inject db session"}

    logger.info(f"db session successfully injected, dummy query passed")
    return {"message": "db session successfully injected, dummy query passed"}

# test route for Pydantic-validated response schema and request body
@router.post("/third-test", response_model=TestResponse)
async def third_test(
    request: TestRequest,
    response: Response
):
    # logging to test request body parsing
    logger.info(f"Third test route, request: {request}")
    if request.requested_num is None:
        logger.info("requested_num is None")
    elif request.requested_num > 0:
        logger.info(f"requested_num is positive")
    elif request.requested_num < 0:
        logger.info(f"requested_num is negative")
    else:
        logger.info(f"requested_num is 0")
    
    # NOTE: FastAPI allows returning the actual pydantic model obj.
    response_obj = TestResponse(
        message=f"the requested msg is: {request.request_msg}",
        list_msg=["this", "is", "a", "list", "of", "strings"]
    )
    # adjust the Response status directly
    response.status_code = status.HTTP_200_OK
    return response_obj

# test llm observation tracking
@router.post("/fourth-test")
async def start_observation_tracking(
    poll_interval: float = Query(default=1.0, gt=0.0),
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker)
):
    try:
        # tracking_task = observation_tracker.start_background_tracking(poll_interval=poll_interval)
        # logger.info(f"Observation tracking task started: {tracking_task}")
        await observation_tracker.create_test_observation()
        return {"message": "successfully tested llm observation!"}
    except Exception as e:
        logger.error(f"Error starting observation tracking task: {e}")
        return {"message": f"Error starting observation tracking task: {e}"}, 500
