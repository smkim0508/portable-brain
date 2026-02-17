# tool calling declarations for droidrun functions
from google import genai
from google.genai import types

# import droidrun helpers for execution
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient

# Define a function that the model can call to execute commands on device via droidrun
# NOTE: the declaration telling LLM what/how to use this external function
droidrun_execution_declaration = {
    "name": "execute_command",
    "description": "Execute a natural language command on the user's Android device. The command should be a clear, enriched instruction describing the action to perform on the phone.",
    "parameters": {
        "type": "object",
        "properties": {
            "enriched_command": {
                "type": "string",
                "description": "Natural language command to execute on the device. Should be specific and actionable, e.g. 'Open Messages app, send SMS to Kevin Chen (+1-234-567-8900) with message about dinner'",
            },
            "reasoning": {
                "type": "boolean",
                "description": "Whether to enable step-by-step reasoning for complex commands. Defaults to false.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum time in seconds to wait for command execution. Defaults to 120.",
            },
        },
        "required": ["enriched_command"],
    },
}
