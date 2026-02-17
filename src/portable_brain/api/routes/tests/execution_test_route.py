# test route to execute natural language queries on device via droidrun

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from portable_brain.common.logging.logger import logger
from portable_brain.core.dependencies import get_tool_calling_agent
from portable_brain.agent_service.core_agent.tool_calling_agent import ToolCallingAgent

router = APIRouter(prefix="/execution-test", tags=["Tests"])

class ToolCallRequest(BaseModel):
    user_prompt: str

@router.post("/tool-call")
async def test_tool_call(
    request: ToolCallRequest,
    tool_calling_agent: ToolCallingAgent = Depends(get_tool_calling_agent)
):
    """
    Test route: Gemini tool-calls DroidRun's execute_command with a custom user prompt.
    """
    result = await tool_calling_agent.test_tool_call(request.user_prompt)
    logger.info(f"Tool call test result: {result}")
    return {"result": result}
