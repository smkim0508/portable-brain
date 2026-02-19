# main orchestrator agent, a.k.a. the entrypoint for all agent logic
# bridges between memory context retrieval layer and tool calling execution layer

# main agents
from portable_brain.agent_service.execution_agent.agent import ExecutionAgent
from portable_brain.agent_service.retrieval_agent.agent import RetrievalAgent
# types
from portable_brain.agent_service.common.types.llm_outputs.memory_retrieval_outputs import MemoryRetrievalLLMOutput, RetrievalLogEntry
from portable_brain.agent_service.common.types.llm_outputs.execution_outputs import ExecutionLLMOutput
from portable_brain.agent_service.common.types.orchestration_state import RetrievalState
# logging
from portable_brain.common.logging.logger import logger

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

    async def run(
        self,
        user_request: str,
        max_iterations: int = 3,
        execution_agent_max_turns: int = 5,
        retrieval_agent_max_turns: int = 5
    ) -> ExecutionLLMOutput:
        """
        Main orchestration loop: retrieve context -> execute -> re-retrieve on failure.
        Returns the final ExecutionLLMOutput (success or last failed attempt).
        - Maintains a cumulative retrieval log and state for re-retrieval.
        """
        # track cumulative retrieval log across iterations
        all_previous_queries: list[RetrievalLogEntry] = []

        # 1) initial retrieval
        retrieval_raw = await self.retrieval_agent.test_retrieve(user_request) # TODO: update helper method after retrieval agent implementation
        retrieval_result = self._parse_retrieval(retrieval_raw)
        if retrieval_result is not None:
            all_previous_queries.extend(retrieval_result.retrieval_log)
            context = retrieval_result.context_summary
        else:
            context = str(retrieval_raw)

        for iteration in range(max_iterations):
            # 2) execute with retrieved context
            execution_raw = await self.execution_agent.execute_command(
                user_request=user_request,
                context=context,
                max_turns=execution_agent_max_turns
            )
            execution_result = self._parse_execution(execution_raw)

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
            retrieval_raw = await self.retrieval_agent.test_retrieve(re_retrieval_prompt, max_turns=retrieval_agent_max_turns)
            retrieval_result = self._parse_retrieval(retrieval_raw)
            if retrieval_result is not None:
                all_previous_queries.extend(retrieval_result.retrieval_log)
                context = retrieval_result.context_summary
            else:
                context = str(retrieval_raw)

        # exhausted all iterations, return last execution result
        return execution_result

    def _parse_retrieval(self, raw) -> MemoryRetrievalLLMOutput | None:
        """Parse retrieval output, returning None if validation failed and raw text was returned."""
        if isinstance(raw, MemoryRetrievalLLMOutput):
            return raw
        logger.warning(f"Retrieval agent returned raw text (validation failed), proceeding with raw context: {str(raw)}")
        return None

    def _parse_execution(self, raw) -> ExecutionLLMOutput:
        """Parse execution output, wrapping raw text as a failed result if validation failed."""
        if isinstance(raw, ExecutionLLMOutput):
            return raw
        logger.warning(f"Execution agent returned raw text (validation failed): {str(raw)}")
        return ExecutionLLMOutput(
            success=False,
            result_summary=str(raw),
            failure_reason="Execution agent returned unstructured response",
            missing_information=None,
        )
