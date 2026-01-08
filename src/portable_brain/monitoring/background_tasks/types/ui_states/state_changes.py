from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from datetime import datetime, timezone, timedelta
import time
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType
from portable_brain.monitoring.background_tasks.types.ui_states.ui_state import UIState

class StateChangeSource(str, Enum):
    """
    Defines the souce of UI state change.
    Currently supports observations and user-given commands.
    NOTE: the background observation trakcer should only log observations
    - This propagates to the recorded Actions.
    """
    OBSERVATION = "observation"
    COMMAND = "command"

class UIStateChange(BaseModel):
    """
    UI Change DTO for low-level monitoring.
    Stores the before and after change states, and additional metadata.
    """
    timestamp: datetime
    change_type: StateChangeType
    before: UIState
    after: UIState
    source: StateChangeSource
