# main execution agent and helpers

# import clients
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient

# import tool calling declarations
from portable_brain.agent_service.common.tool_calling_declarations.droidrun_execution import droidrun_execution_declaration
# import system prompts
from portable_brain.agent_service.common.system_prompts.device_execution_prompts import *
# LLM output schema
from portable_brain.agent_service.common.types.llm_outputs.execution_outputs import ExecutionLLMOutput

class ExecutionAgent():
    """
    Main execution agent that tool calls to DroidRun client to execute commands on device.
    Goals:
        - Decide whether a real-world/tool action is necessary
        - Convert user request + retrieved context into an executable command
        - Invoke external tools (e.g., DroidRun) to perform actions
        - Capture structured results (success, outputs, reasoning)
        - Diagnose failures and infer likely missing information
        - Produce metadata about attempted actions (command, errors, failure cause)
        - Request additional retrieval when execution cannot succeed yet
        - Retry execution after new context is obtained
        - Return final answer or confirmation once action succeeds

    Receives memory context and executes commands on device via tool calls to droidrun.
    NOTE: initialized during lifespan and reused in the service lifecycle.
    """
    def __init__(self, droidrun_client: DroidRunClient, gemini_llm_client: AsyncGenAITypedClient):
        self.droidrun_client = droidrun_client
        self.llm_client = gemini_llm_client # NOTE: for now, this llm client must be the gemini client (not dispatcher) to allow atool_call() method
    
    # test helper to connect with droidrun
    # NOTE: very minimal system prompt, no memory context
    async def test_tool_call(self, user_prompt: str):
        test_system_prompt = f"""
        You are an AI agent that controls the user's Android phone.
        You have access to the execute_command tool which lets you perform actions on the device.        
        You MUST use the execute_command tool to carry out any request about the device. 
        Always use the tool first, then respond with the result.
        """

        return await self.llm_client.atool_call(
            system_prompt=test_system_prompt,
            user_prompt=user_prompt,
            function_declarations=[droidrun_execution_declaration],
            tool_executors={"execute_command": self.droidrun_client.execute_command},
            max_turns=5
        )
    
    # baseline test: full prompt structure but no augmented context
    async def mocked_execute_command(self, user_request: str, max_turns: int = 5):
        return await self.llm_client.atool_call(
            system_prompt=DeviceExecutionPrompts.direct_execution_system_prompt,
            user_prompt=user_request,
            function_declarations=[droidrun_execution_declaration],
            tool_executors={"execute_command": self.droidrun_client.execute_command},
            response_model=ExecutionLLMOutput,
            max_turns=max_turns,
        )
    
    async def execute_command(self, user_request: str, context: str, max_turns: int = 5):
        """
        Main tool calling method to execute commands on device, with relevant memory context

        Args: context is given as a plain natural language string, alongside the original user request.
        """
        user_prompt = user_request + "\n\n Context: \n" + context
        # or, make a new semantically enriched user prompt via LLM pass (TBD)

        return await self.llm_client.atool_call(
            system_prompt=DeviceExecutionPrompts.device_execution_system_prompt,
            user_prompt=user_prompt,
            function_declarations=[droidrun_execution_declaration],
            tool_executors={"execute_command": self.droidrun_client.execute_command},
            response_model=ExecutionLLMOutput,
            max_turns=max_turns,
        )
