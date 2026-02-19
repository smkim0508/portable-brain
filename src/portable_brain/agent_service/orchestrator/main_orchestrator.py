# main orchestrator agent, a.k.a. the entrypoint for all agent logic
# bridges between memory context retrieval layer and tool calling execution layer

# main agents
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
# types
from portable_brain.agent_service.common.types.llm_outputs.memory_retrieval_outputs import MemoryRetrievalLLMOutput, RetrievalLogEntry
from portable_brain.agent_service.common.types.llm_outputs.execution_outputs import ExecutionLLMOutput
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

    # TODO: later, make orchestration settings to hold params like max loop
    async def run(self, user_request: str, max_iterations: int = 3) -> ExecutionLLMOutput:
        """
        Main orchestration loop: retrieve context -> execute -> re-retrieve on failure.
        Returns the final ExecutionLLMOutput (success or last failed attempt).
        - Maintains a cumulative retrieval log and state for re-retrieval.
        """
        # track cumulative retrieval log across iterations
        all_previous_queries: list[RetrievalLogEntry] = []

        # 1) initial retrieval
        retrieval_raw = await self.retrieval_agent.test_retrieve(user_request) # TODO: update helper method after retrieval agent implementation
        if not isinstance(retrieval_raw, MemoryRetrievalLLMOutput):
            raise RuntimeError(f"Retrieval agent returned unparseable response: {retrieval_raw}")
        retrieval_result: MemoryRetrievalLLMOutput = retrieval_raw
        all_previous_queries.extend(retrieval_result.retrieval_log)

        for iteration in range(max_iterations):
            # 2) execute with retrieved context
            execution_raw = await self.execution_agent.execute_command(
                user_request=user_request,
                context=retrieval_result.context_summary,
            )
            if not isinstance(execution_raw, ExecutionLLMOutput):
                raise RuntimeError(f"Execution agent returned unparseable response: {execution_raw}")
            execution_result: ExecutionLLMOutput = execution_raw

            # 3) check success
            if execution_result.success:
                return execution_result

            # 4) build retrieval state for re-retrieval
            self.retrieval_state = RetrievalState(
                iteration=iteration + 1,
                previous_queries=all_previous_queries,
                execution_failure_reason=execution_result.failure_reason or "Unknown failure",
                missing_information=execution_result.missing_information or "Unknown",
            )

            # 5) re-retrieve with state appended to user prompt
            re_retrieval_prompt = user_request + "\n\nretrieval_state:\n" + self.retrieval_state.model_dump_json()
            retrieval_raw = await self.retrieval_agent.test_retrieve(re_retrieval_prompt)
            if not isinstance(retrieval_raw, MemoryRetrievalLLMOutput):
                raise RuntimeError(f"Re-retrieval agent returned unparseable response: {retrieval_raw}")
            retrieval_result = retrieval_raw
            all_previous_queries.extend(retrieval_result.retrieval_log)

        # exhausted all iterations, return last execution result
        return execution_result
