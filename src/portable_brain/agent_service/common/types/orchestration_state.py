
from pydantic import BaseModel, Field
from portable_brain.agent_service.common.types.llm_outputs.memory_retrieval_outputs import RetrievalLogEntry

class RetrievalState(BaseModel):
    """
    Cumulative state passed to the retrieval agent on re-retrieval after execution failure.
    Tracks what has already been tried so the agent can avoid redundant queries.
    """
    iteration: int = Field(description="Current re-retrieval attempt number. Starts at 1 for the first re-retrieval.")
    previous_queries: list[RetrievalLogEntry] = Field(description="All tool calls from prior retrieval turns.")
    execution_failure_reason: str = Field(description="Why the execution agent's previous attempt failed.")
    missing_information: str = Field(description="Execution agent's best guess at what information is still needed.")
