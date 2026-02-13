# test route for embedding-related functionality

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

router = APIRouter(prefix="/embedding-test", tags=["Tests"])


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
