# test route for router + dependency injection

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

router = APIRouter(prefix="/test", tags=["Tests"])

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

@router.post("/test-text-embedding")
async def test_text_embedding(
    request: TestEmbeddingRequest,
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
    embedding_client: TypedTextEmbeddingClient = Depends(get_gemini_text_embedding_client),
    main_db_engine: AsyncEngine = Depends(get_main_db_engine)
):
    # parse request
    observation_id = request.observation_id
    test_embedding_text = request.embedding_text
    # attempt to force tracker to generate and save embedding using convenience helper
    try:
        await observation_tracker.embedding_generator.generate_and_save_embedding(observation_id=observation_id, observation_text=request.embedding_text)
        return {"message": "successfully tested text embedding!"}
    except Exception as e:
        logger.error(f"Error testing text embedding: {e}")
        return {"message": f"Error testing text embedding: {e}"}, 500

@router.post("/find-similar-embedding", response_model=SimilarEmbeddingResponse)
async def find_similar_embedding(
    request: SimilarEmbeddingRequest,
    embedding_client: TypedTextEmbeddingClient = Depends(get_gemini_text_embedding_client),
    main_db_engine: AsyncEngine = Depends(get_main_db_engine)
):
    try:
        # embed the target text
        target_embeddings = await embedding_client.aembed_text([request.target_text])
        target_vector = target_embeddings[0]

        # find closest match in db by cosine distance
        results = await find_similar_embeddings(
            query_vector=target_vector,
            limit=1,
            main_db_engine=main_db_engine,
            distance_metric="cosine"
        )

        if not results:
            raise HTTPException(status_code=404, detail="No embeddings found in the database.")

        closest_record, distance = results[0]

        logger.info(f"Found closest embedding for '{request.target_text}' with distance {distance:.4f}")
        return SimilarEmbeddingResponse(
            closest_text=closest_record.observation_text,
            cosine_similarity_distance=distance,
            target_embedding=target_vector[:5],
            closest_embedding=closest_record.embedding_vector[:5],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error finding similar embedding: {e}")
