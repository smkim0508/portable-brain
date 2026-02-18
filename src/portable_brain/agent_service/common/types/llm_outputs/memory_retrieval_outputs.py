# output types for memory retrieval agent

from pydantic import BaseModel, Field

class RetrievalLogEntry(BaseModel):
    """A single tool call record for the retrieval log."""
    tool: str = Field(description="Name of the memory retriever tool called.")
    params: dict = Field(description="Parameters passed to the tool call.")
    result_summary: str = Field(description="Brief summary of what the tool returned.")

class RetrievalState(BaseModel):
    """
    Cumulative state passed to the retrieval agent on re-retrieval after execution failure.
    Tracks what has already been tried so the agent can avoid redundant queries.
    """
    iteration: int = Field(description="Current re-retrieval attempt number. Starts at 1 for the first re-retrieval.")
    previous_queries: list[RetrievalLogEntry] = Field(description="All tool calls from prior retrieval turns.")
    execution_failure_reason: str = Field(description="Why the execution agent's previous attempt failed.")
    missing_information: str = Field(description="Execution agent's best guess at what information is still needed.")

class MemoryRetrievalOutput(BaseModel):
    """
    Structured output from the retrieval agent.
    Consumed directly by the execution agent to build enriched commands.
    """
    context_summary: str = Field(description="Natural language paragraph of all relevant facts retrieved from memory.")
    inferred_intent: str = Field(description="Single clear sentence describing the user's resolved intent.")
    reasoning: str = Field(description="Step-by-step reasoning trace for debugging and transparency.")
    unresolved: list[str] = Field(default_factory=list, description="Specific pieces of information not found in memory. Empty if everything is resolved.")
    retrieval_log: list[RetrievalLogEntry] = Field(default_factory=list, description="Tool calls made in this turn, used by future re-retrieval turns to avoid redundancy.")
