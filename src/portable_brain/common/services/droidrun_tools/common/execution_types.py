from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from portable_brain.monitoring.background_tasks.types.ui_states.ui_state import UIState
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType

class RawExecutionResult(BaseModel):
    """
    Canonical representation of a bare, single DroidRun command execution result.
    Records the command, outcome, and before/after device state.
    - Thinly wraps around DroidRun's ResultEvent object fields with minimal metadata: timestamp and command
    """
    timestamp: datetime = datetime.now()
    command: str
    success: bool
    reason: Optional[str] = None
    steps: int
    structured_output: Optional[BaseModel] = None

class ExecutionResult(BaseModel):
    """
    Canonical representation of a single DroidRun command execution result.
    Records the command, outcome, and before/after device state.
    - Wraps around DroidRun's ResultEvent object fields with enriched metadata like command + UI state diffs.
    """
    timestamp: datetime = datetime.now()
    command: str
    success: bool
    reason: Optional[str] = None
    steps: int
    state_before: UIState
    state_after: UIState
    change_type: StateChangeType
    structured_output: Optional[BaseModel] = None
