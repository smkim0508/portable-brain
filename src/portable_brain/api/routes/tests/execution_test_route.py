# test route to execute natural language queries on device via droidrun

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from portable_brain.common.logging.logger import logger

# agents
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
from portable_brain.agent_service.orchestrator.main_orchestrator import MainOrchestrator

# dependencies
from portable_brain.core.dependencies import (
    get_execution_agent,
    get_retrieval_agent
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
    Test route: Gemini tool-calls DroidRun's execute_command with a custom user prompt.
    """
    result = await tool_calling_agent.test_tool_call(request.user_request)
    logger.info(f"Tool call test result: {result}")
    return {"result": result}

@router.post("/rag-execution-test")
async def rag_execution_test(
    request: ToolCallRequest, # TODO update this request param
    execution_agent: ExecutionAgent = Depends(get_execution_agent),
    retrieval_agent: RetrievalAgent = Depends(get_retrieval_agent)
):
    """
    Test route: Gemini tool-calls DroidRun's execute_command with a custom user prompt.
    """
    main_orchestrator = MainOrchestrator(execution_agent, retrieval_agent)
    result = await main_orchestrator.run(request.user_request)
    logger.info(f"RAG execution test result: {result}")
    return {"result": result}

# TODO: finish building this test api and the baseline
