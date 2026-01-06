from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum

class ChangeType(str, Enum):
    """
    Low-level classifications for different UI changes.
    Used to infer high-level actions.
    TODO: add more change types
    """
    APP_SWITCH = "app_switch"
    SCREEN_CHANGE = "screen_change"
    MAJOR_LAYOUT_CHANGE = "major_layout_change"
    MINOR_LAYOUT_CHANGE = "minor_layout_change"
    SCREEN_NAVIGATION = "screen_navigation"
    CONTENT_NAVIGATION = "content_navigation"
    TEXT_INPUT = "text_input"
