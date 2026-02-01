# observation DTOs
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Union
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

class MemoryType(str, Enum):
    """
    Classification for which memory data structure this observation is associated with.
    - Default classification is current session.
    """
    LONG_TERM_PEOPLE = "long_term_people" # inter-personal relationships
    LONG_TERM_PREFERENCES = "long_term_preferences" # user preferences, or user-object relationships
    SHORT_TERM_PREFERENCES = "short_term_preferences" # short term user preferences
    SHORT_TERM_CONTENT = "short_term_content" # short term content context storage
    # for now, current session is simply metadata pulled without memory
    CURRENT_SESSION = "current_session" # current session context
    # TODO: open to expansion

class ObservationBase(BaseModel):
    """
    Base class for all observations.
    Holds common metadata across observations.
    """
    id: str # unique identifier
    memory_type: MemoryType # which memory this observation is associated with
    importance: float # node weight
    created_at: datetime # recency calculation

class LongTermPeopleObservation(ObservationBase):
    """
    Observation for inter-personal relationships, which is a long term memory.
    - e.g. relationship between me and another person in contacts
    NOTE: any people related observation should be a long term memory
    """
    memory_type: MemoryType = MemoryType.LONG_TERM_PEOPLE
    target_id: str # id of the target person, as a unique identifier
    edge: str # semantic classification of the node type w.r.t. target
    node: str # semantic description of the relationship/observation
    primary_communication_channel: str

class ShortTermPreferencesObservation(ObservationBase):
    """
    Observation for SHORT TERM user preferences.
    """
    memory_type: MemoryType = MemoryType.SHORT_TERM_PREFERENCES
    source_id: str # id of the source object (e.g. app) that this prefernece is relevant to, as a unique identifier
    edge: str # semantic classification of the node type w.r.t. target
    node: str # semantic description of the relationship/observation
    recurrence: int # number of occurrences that this preference is recorded

class LongTermPreferencesObservation(ObservationBase):
    """
    Observation for LONG TERM user preferences.
    - e.g. recurring pattern of application usage (like email -> slack)
    """
    memory_type: MemoryType = MemoryType.LONG_TERM_PREFERENCES
    source_id: str # id of the target object (e.g. app) that this preference is relevant to, as a unique identifier
    edge: str # semantic classification of the node type w.r.t. target
    node: str # semantic description of the relationship/observation
    recurrence: int # number of occurrences that this preference is recorded

class ShortTermContentObservation(ObservationBase):
    """
    Observation for short term content context, which is a short term memory.
    - e.g. recently viewed documents or media.
    NOTE: content is only a short term memory, since we don't track media over long period.
    """
    memory_type: MemoryType = MemoryType.SHORT_TERM_CONTENT
    source_id: str # unique identifier of the source of the content
    content_id: str # unique identifier of the content 
    node: str # semantic description of the content
    
# class Observation(BaseModel):
#     """
#     High-level inference of user behavior, derived from a union of low-level actions.
#     Each observation holds semantic inference and additional metadata.
#     NOTE: shared across agent-executed action observation and user-inferred observations
#     """
#     description: str # semantic inference
#     importance: float # temporarily defined here
#     timestamp: datetime
#     behavior_type: BehaviorType # type of behavior, like recurring w.r.t. time/sequence of actions

Observation = Union[
    LongTermPeopleObservation,
    LongTermPreferencesObservation,
    ShortTermPreferencesObservation,
    ShortTermContentObservation
    # TODO: add more
]
