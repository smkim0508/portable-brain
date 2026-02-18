# main execution agent and helpers

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
    
    TODO: implement helpers, merge w/ ToolCallingAgent
    """
    def __init__(self):
        pass
