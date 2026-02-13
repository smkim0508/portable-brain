# semantic observation classification LLM responses
from pydantic import BaseModel
from portable_brain.monitoring.background_tasks.types.observation.observations import MemoryType

class ClassifyObservationLLMResponse(BaseModel):
    """
    Observation LLM response schema for classifying an observation node into a MemoryType.
    NOTE: The LLM outputs only the MemoryType classification and its reasoning, not a full Observation object.
    """
    classification_result: MemoryType
    reasoning: str
