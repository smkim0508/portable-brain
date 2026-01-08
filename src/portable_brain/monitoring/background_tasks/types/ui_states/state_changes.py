from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from datetime import datetime, timezone, timedelta
import time
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType

class UIStateChange(BaseModel):
    """
    UI Change DTO for low-level monitoring.
    """
    timestamp: datetime
    change_type: StateChangeType
    
    
