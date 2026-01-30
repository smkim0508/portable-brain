# observation DTOs
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from datetime import datetime

class BehaviorType(str, Enum):
    """
    Classification for common behavior types, helps to identify node structure in memory.
    """
    RECURRING_TIME = "recurring_time"
    RECURRING_SEQUENTIAL = "recurring_sequential"
    TARGET_CLARIFICATION = "target_classification"
    UNKNOWN = "unknown"
    # TODO: add more

class Observation(BaseModel):
    """
    High-level inference of user behavior, derived from a union of low-level actions.
    Each observation holds semantic inference and additional metadata.
    NOTE: shared across agent-executed action observation and user-inferred observations
    """
    description: str # semantic inference
    importance: float # temporarily defined here
    timestamp: datetime
    behavior_type: BehaviorType # type of behavior, like recurring w.r.t. time/sequence of actions
