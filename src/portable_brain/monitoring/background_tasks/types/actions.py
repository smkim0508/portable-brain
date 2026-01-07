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

# TODO: for now, experiment with mostly text messages across three applications, then expand later

# NOTE: the convention for naming should follow src/dst or actor/target depending on relationship
class AppSwitchAction(GenericActionBase):
    """
    Action for switching between apps.
    """
    type: Literal[GenericActionType.APP_SWITCH] = GenericActionType.APP_SWITCH
    src_package: str
    dst_package: str
    src_activity: Optional[str] = None
    dst_activity: Optional[str] = None

class InstagramMessageSentAction(InstagramActionBase):
    """
    Action for sending text message on Instagram.
    """
    # NOTE: should log entire stream of messages in a single session, not every single message
    type: Literal[InstagramActionType.MESSAGE_SENT] = InstagramActionType.MESSAGE_SENT
    target_username: str
    message_summary: Optional[str] = None

class InstagramPostLikedAction(InstagramActionBase):
    """
    Action for liking a post on Instagram.
    """
    type: Literal[InstagramActionType.POST_LIKED] = InstagramActionType.POST_LIKED
    target_username: str
    post_description: Optional[str] = None

class WhatsAppMessageSentAction(WhatsAppActionBase):
    """
    Action for sending text message on WhatsApp.
    """
    # NOTE: should log entire stream of messages in a single session, not every single message
    type: Literal[WhatsAppActionType.MESSAGE_SENT] = WhatsAppActionType.MESSAGE_SENT
    target_name: str
    message_summary: Optional[str] = None

class SlackMessageSentAction(SlackActionBase):
    """
    Action for sending text message on Slack.
    """
    # NOTE: should log entire stream of messages in a single session, not every single message
    type: Literal[SlackActionType.MESSAGE_SENT] = SlackActionType.MESSAGE_SENT
    thread_name: Optional[str] = None # overrides, assumed None
    is_dm: bool = False # overrides, default assumed to be false
    target_name: Optional[str] = None # only if message is a dm
    message_summary: Optional[str] = None

# NOTE: uses Union to represent all possible action types without losing Pydantic schema
Action = Union[
    # Generic actions
    AppSwitchAction,
    # Instagram actions
    InstagramMessageSentAction,
    InstagramPostLikedAction,
    # WhatsApp actions
    WhatsAppMessageSentAction,
    # Slack actions
    SlackMessageSentAction
    # more to be added...
]

def format_action(action_type: ActionType):
    """
    May be used to structure actions in canonical format
    """
    pass
