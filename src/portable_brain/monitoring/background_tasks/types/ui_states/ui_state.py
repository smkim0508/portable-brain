from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from datetime import datetime, timezone, timedelta
import time

class UIActivity(BaseModel):
    """
    Canonical representation of a single Android UI activity.
    NOTE:
    - A single package may have multiple activities.
    - A single activity may contain multiple UI components.
    """
    activity: str

class UIState(BaseModel):
    """
    Canonical representation of Android UI State.
    - A11Y tree output translated into portable format.
    TODO: think about what states to record here!
    """
    state_id: str
    package: str # which app am I on?
    activity: UIActivity # which screen within the app am I on?
    ui_elements: list # TODO: list of ?
    focused_element: Optional[int] = None # the currently selected element, by idx
    formatted_text: str # a human-readable description of the UI state NOTE: used by observation inferencer
    raw_tree: Optional[dict] = None # NOTE: need a way to reliably fetch app-specific metadata like username
    raw_tree_hash: Optional[str] = None # optionally, to keep track of any additional info
