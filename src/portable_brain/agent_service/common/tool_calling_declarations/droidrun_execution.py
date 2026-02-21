# tool calling declarations for droidrun functions

# Define a function that the model can call to execute commands on device via droidrun
# NOTE: the declaration telling LLM what/how to use this external function
droidrun_execution_declaration = {
    "name": "execute_command",
    "description": (
        "Execute a natural language command on the user's Android device. "
        "The command should be a clear, enriched instruction describing the action to perform on the phone. "
        "Returns a result object with: success (bool), reason (str — explanation or answer from the device agent), "
        "steps (int — number of steps taken), command (str — the command that was executed), "
        "and timestamp (datetime)."
    ),
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
