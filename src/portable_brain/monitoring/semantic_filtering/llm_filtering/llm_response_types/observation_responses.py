# semantic observation LLM responses
from pydantic import BaseModel
from typing import Optional
from enum import Enum

class TestObservationLLMResponse(BaseModel):
    """
    Test LLM response schema for creating a new observation.
    """
    observation_edge: str # relationship between source and target
    observation_node: str # semantic meaning of this observation
    reasoning: str

class NewObservationLLMResponse(BaseModel):
    """
    LLM response schema for creating a new observation.
    NOTE: the observation edge will be inferenced later when retrieving similar observations.
    """
    observation_node: Optional[str] # semantic meaning of this observation
    reasoning: str # step-by-step reasoning

class UpdatedObservationLLMResponse(BaseModel):
    """
    LLM response schema for updating an observation.
    If no meaningful observation can be inferred, should mark is_updated=False and return None for node.
    """
    updated_observation_node: Optional[str] # semantic meaning of this observation
    is_updated: bool
    reasoning: str
