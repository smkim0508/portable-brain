from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from portable_brain.monitoring.background_tasks.types.action_bases import (
    ActionBase,
    GenericActionBase,
    InstagramActionBase,
    WhatsAppActionBase,
    SlackActionBase
)
from portable_brain.monitoring.background_tasks.types.action_types import (
    ActionType,
    GenericActionType,
    InstagramActionType,
    WhatsAppActionType,
    SlackActionType
)

class ActionSource(str, Enum):
    """
    Defines the souce of action.
    Currently supports observations and user-given commands.
    NOTE: the background observation trakcer should only log observations
    """
    OBSERVATION = "observation"
    COMMAND = "command"

# specific actions, as defined by the types above
# NOTE: each action inherits shared metadata from ActionBase class
class AppSwitchAction(GenericActionBase):
    """
    User switched between apps.
    """
    type: Literal[GenericActionType.APP_SWITCH] = GenericActionType.APP_SWITCH
    from_package: str
    to_package: str
    from_activity: Optional[str] = None
    to_activity: Optional[str] = None

class InstagramMessageSentAction(InstagramActionBase):
    # NOTE: should log entire stream of messages in a single session, not every single message
    type: Literal[InstagramActionType.MESSAGE_SENT] = InstagramActionType.MESSAGE_SENT
    sender_username: str
    recipient_username: str
    message_summary: str

# NOTE: uses Union to represent all possible action types without losing Pydantic schema
Action = Union[
    AppSwitchAction,
    InstagramMessageSentAction
    # more to be added...
]

def format_action(action_type: ActionType):
    """
    May be used to structure actions in canonical format
    """
    pass
