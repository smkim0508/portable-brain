# test route to execute natural language queries on device via droidrun

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from portable_brain.common.logging.logger import logger

# agents
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
from portable_brain.agent_service.orchestrator.main_orchestrator import MainOrchestrator
# droidrun
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient

# dependencies
from portable_brain.core.dependencies import (
    get_execution_agent,
    get_retrieval_agent,
    get_droidrun_client
)

# request models
from portable_brain.api.request_models.tests import ToolCallRequest

router = APIRouter(prefix="/execution-test", tags=["Tests"])

@router.post("/tool-call")
async def test_tool_call(
    request: ToolCallRequest,
    tool_calling_agent: ExecutionAgent = Depends(get_execution_agent)
):
    """
    Test route: Gemini tool-calls DroidRun's execute_command with a custom user request.
    """
    result = await tool_calling_agent.test_tool_call(request.user_request)
    logger.info(f"Tool call test result: {result}")
    return {"result": result}

# NOTE: below are three sets of execution tests for benchmarking
# Evaluates the contribution of each component: execution agent and retrieval agent (+ orchestrator)
@router.post("/orchestrated-execution-test")
async def rag_execution_test(
    request: ToolCallRequest,
    execution_agent: ExecutionAgent = Depends(get_execution_agent),
    retrieval_agent: RetrievalAgent = Depends(get_retrieval_agent)
):
    """
    Tests the main orchestration logic and full retrieval-execution loop.
    - Takes custom user request to be parsed and executed via augmented context.

    - yes semantic enrichment of user request
    - yes augmented context
    """
    main_orchestrator = MainOrchestrator(execution_agent, retrieval_agent)
    result = await main_orchestrator.run(request.user_request)
    logger.info(f"RAG execution test result: {result}")
    return {"result": result}

@router.post("/no-context-execution-test")
async def direct_execution_test(
    request: ToolCallRequest,
    execution_agent: ExecutionAgent = Depends(get_execution_agent)
):
    """
    Tests JUST the baseline droidrun's execution without any augmented context.
    - yes semantic enrichment of user request
    - no augmented context
    """
    
    result = await execution_agent.mocked_execute_command(request.user_request)
    logger.info(f"No augmented context execution test result: {result}")
    return {"result": result}

@router.post("/direct-droidrun-execution-test")
async def direct_droidrun_execution_test(
    request: ToolCallRequest,
    droidrun_client: DroidRunClient = Depends(get_droidrun_client)
):
    """
    Test to directly execute a command on device via droidrun client, bypasses execution agent.
    - no semantic enrichment of user request
    - no augmented context
    """
    
    result = await droidrun_client.execute_command(request.user_request)
    logger.info(f"Direct DroidRun execution test result: {result}")
    return {"result": result}
