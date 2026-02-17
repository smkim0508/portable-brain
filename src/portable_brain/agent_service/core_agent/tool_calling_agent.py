# main tool calling agent, connects to droidrun

# import clients
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient

# import tool calling declarations
from portable_brain.common.services.llm_service.tool_calling.gemini.droidrun_tool_declaration import droidrun_execution_declaration

class ToolCallingAgent():
    """
    Main agent to handle tool calling and execution.
    Receives memory context and executes commands on device via tool calls to droidrun.

    NOTE: no repository intialized yet.
    """
    def __init__(self, droidrun_client: DroidRunClient, gemini_llm_client: AsyncGenAITypedClient):
        self.droidrun_client = droidrun_client
        self.llm_client = gemini_llm_client # NOTE: for now, this llm client must be the gemini client (not dispatcher) to allow atool_call() method
    
    # test helper to connect with droidrun
    def test_tool_call(self, user_prompt: str):
        test_system_prompt = (
            "You are an AI agent that controls the user's Android phone. "
            "You have access to the execute_command tool which lets you perform actions on the device. "
            "You MUST use the execute_command tool to carry out any request about the device. "
            "Always use the tool first, then respond with the result."
        )

        return self.llm_client.atool_call(
            system_prompt=test_system_prompt,
            user_prompt=user_prompt,
            function_declarations=[droidrun_execution_declaration],
            tool_executors={"execute_command": self.droidrun_client.execute_command},
            max_turns=5
        )
