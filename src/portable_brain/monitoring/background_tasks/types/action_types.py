from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum

class ActionSource(str, Enum):
    """
    Defines the souce of action.
    Currently supports observations and user-given commands.
    """
    OBSERVATION = "observation"
    COMMAND = "command"

class ActionType(str, Enum):
    """
    High-level action types inferred from low-level UI changes.
    TODO: add more actions
    """
    APP_SWITCH = "app_switch"
    MESSAGE_SENT = "message_sent"
    MEDIA_SHARED = "media_shared"

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

class ActionBase(BaseModel):
    """
    High-level inference for what "action" was done by the user.
    - Inferred based on low-level UI change signals.

    This serves as the base model for all actions with shared metadata.
    - Action classes inherit from this base with specific metadata.
    """
    timestamp: str
    description: Optional[str] = None # Human-readable description
    source: ActionSource
    package: str

class GenericActionBase(ActionBase):
    """
    Generic app-agnostic action.
    No addtional metadata is recorded for now.
    """
    pass

# specific actions, as defined by the types above
# NOTE: each action inherits shared metadata from ActionBase class
class AppSwitchAction(ActionBase):
    """
    User switched between apps.
    """
    type: Literal[ActionType.APP_SWITCH] = ActionType.APP_SWITCH
    from_package: str
    to_package: str
    from_activity: Optional[str] = None
    to_activity: Optional[str] = None


# NOTE: uses Union to represent all possible action types without losing Pydantic schema
Action = Union[
    AppSwitchAction,
    # more to be added...
]

def format_action(action_type: ActionType):
    """
    May be used to structure actions in canonical format
    """
    pass
