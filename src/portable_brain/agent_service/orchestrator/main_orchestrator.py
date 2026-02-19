# main orchestrator agent, a.k.a. the entrypoint for all agent logic
# should bridge between memory context retrieval layer and tool calling execution layer

# holds the tool calling agent and memory context retrieval layer
# initialized request-scope, holds necessary metadata during execution loop

# main agents
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
# types
from portable_brain.agent_service.common.types.llm_outputs.memory_retrieval_outputs import MemoryRetrievalLLMOutput
from portable_brain.agent_service.common.types.orchestration_state import RetrievalState

class MainOrchestrator():
    """
    Main orchestration layer that loops between the retrieval agent and the execution agent.
    - The orchestrator is initialize request-scope, and holds necessary metadata during a single request loop.
    """
    def __init__(self, execution_agent: ExecutionAgent, retrieval_agent: RetrievalAgent):
        self.execution_agent: ExecutionAgent = execution_agent
        self.retrieval_agent: RetrievalAgent = retrieval_agent
        # define any state variables/metadata
        self.retrieval_state: RetrievalState

    # TODO: build this loop
