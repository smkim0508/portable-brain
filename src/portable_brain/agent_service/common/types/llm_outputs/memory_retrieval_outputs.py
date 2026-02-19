# output types for memory retrieval agent

from pydantic import BaseModel, Field

class RetrievalLogEntry(BaseModel):
    """A single tool call record for the retrieval log."""
    tool: str = Field(description="Name of the memory retriever tool called.")
    params: dict = Field(description="Parameters passed to the tool call.")
    result_summary: str = Field(description="Brief summary of what the tool returned.")

class MemoryRetrievalLLMOutput(BaseModel):
    """
    Structured output from the retrieval agent.
    Consumed directly by the execution agent to build enriched commands.
    """
    context_summary: str = Field(description="Natural language paragraph of all relevant facts retrieved from memory.")
    inferred_intent: str = Field(description="Single clear sentence describing the user's resolved intent.")
    reasoning: str = Field(description="Step-by-step reasoning trace for debugging and transparency.")
    unresolved: list[str] = Field(default_factory=list, description="Specific pieces of information not found in memory. Empty if everything is resolved.")
    retrieval_log: list[RetrievalLogEntry] = Field(default_factory=list, description="Tool calls made in this turn, used by future re-retrieval turns to avoid redundancy.")
