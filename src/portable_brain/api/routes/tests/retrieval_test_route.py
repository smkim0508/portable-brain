# test route to retrieve relevant memory context from RetrievalAgent
# also tests latency with retrieval from the memory interface layer

import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from portable_brain.common.logging.logger import logger

# agents
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
# droidrun
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient

# dependencies
from portable_brain.core.dependencies import (
    get_retrieval_agent,
    get_droidrun_client,
    get_memory_retriever,
)
from portable_brain.memory.main_retriever import MemoryRetriever

# request models
from portable_brain.api.request_models.tests import ToolCallRequest, SemanticSearchRequest, FindPersonByNameRequest

# settings
from portable_brain.config.app_config import get_service_settings

settings = get_service_settings()

router = APIRouter(prefix="/retrieval-test", tags=["Agent Tests"])

@router.post("/retrieval-test")
async def test_tool_call(
    request: ToolCallRequest,
    retrieval_agent: RetrievalAgent = Depends(get_retrieval_agent)
):
    """
    Test route: retrieves relevant memory context from RetrievalAgent.
    """
    # NOTE: only fetches from text log memory
    result = await retrieval_agent.test_retrieve(request.user_request)
    # logger.info(f"Retrieval test result: {result}")
    return {"result": result}

@router.post("/find-person-by-name")
async def test_find_person_by_name(
    request: FindPersonByNameRequest,
    memory_retriever: MemoryRetriever = Depends(get_memory_retriever),
):
    """
    Test route: calls find_person_by_name directly and logs retrieval latency.
    NOT an agent test exactly, but measures the latency of tool called retrieval method w/ exact name match cache.
    """
    start = time.perf_counter()
    results = await memory_retriever.find_person_by_name(
        name=request.name,
        similarity_threshold=request.similarity_threshold,
        limit=request.limit,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"find_person_by_name '{request.name}' took {elapsed_ms:.2f}ms, {len(results)} result(s)")
    return {"results": results, "elapsed_ms": round(elapsed_ms, 2)}

@router.post("/semantic-search")
async def test_semantic_search(
    request: SemanticSearchRequest,
    memory_retriever: MemoryRetriever = Depends(get_memory_retriever),
):
    """
    Test route: calls find_semantically_similar directly and logs retrieval latency.
    NOT an agent test exactly, but measures the latency of tool called retrieval method w/ caches.
    """
    start = time.perf_counter()
    results = await memory_retriever.find_semantically_similar(
        query=request.query,
        limit=request.limit,
        disable_cache=request.disable_cache,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Semantic search took {elapsed_ms:.2f}ms (cache disabled: {request.disable_cache})")
    return {"results": results, "elapsed_ms": round(elapsed_ms, 2)}
