from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum

class GenericActionType(str, Enum):
    """
    High-level, generic action types inferred from low-level UI changes.
    TODO: add more actions
    """
    APP_SWITCH = "app_switch"

class InstagramActionType(str, Enum):
    """
    High-level, Instagram-specific action types inferred from low-level UI changes.
    TODO: add more actions
    """
    MESSAGE_SENT = "message_sent"
    POST_SHARED = "post_shared"
    REEL_SHARED = "reel_shared"
    POST_LIKED = "post_liked"
    STORY_LIKED = "story_liked"
    STORY_VIEWED = "story_viewed" # should be a less-priority action than story_liked

class WhatsAppActionType(str, Enum):
    """
    High-level, WhatsApp-specific action types inferred from low-level UI changes.
    TODO: add more actions
    """
    MESSAGE_SENT = "message_sent"
    MEDIA_SHARED = "media_shared"
    ATTACHMENT_OPENED = "attachment_opened"

class SlackActionType(str, Enum):
    """
    High-level, Slack-specific action types inferred from low-level UI changes.
    TODO: add more actions
    """
    MESSAGE_SENT = "message_sent"
    MEDIA_SHARED = "media_shared"

# use Union to represent generic Action Types, used in helpers
ActionType = Union[
    GenericActionType,
    InstagramActionType,
    WhatsAppActionType,
    SlackActionType
]
    