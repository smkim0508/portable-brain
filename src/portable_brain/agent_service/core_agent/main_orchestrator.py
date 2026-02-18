# main orchestrator agent, a.k.a. the entrypoint for all agent logic
# should bridge between memory context retrieval layer and tool calling execution layer

# holds the tool calling agent and memory context retrieval layer
# initialized request-scope, holds necessary metadata during execution loop

from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent

class MainOrchestrator():
    """
    Main orchestration layer that loops between the retrieval agent and the execution agent.
    - The orchestrator is initialize request-scope, and holds necessary metadata during a single request loop.
    """
    def __init__(self, execution_agent: ExecutionAgent, retrieval_agent: RetrievalAgent):
        self.execution_agent = execution_agent
        self.retrieval_agent = retrieval_agent
