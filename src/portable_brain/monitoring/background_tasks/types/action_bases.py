from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from portable_brain.monitoring.background_tasks.types.actions import ActionSource

class ActionBase(BaseModel):
    """
    High-level inference for what "action" was done by the user.
    - Inferred based on low-level UI change signals.

    The absolute base class for all actions and shared metadata.
    - App-specific action bases inherit from this base with specific protocols.
    """
    timestamp: str
    description: Optional[str] = None # Human-readable description
    source: ActionSource
    priority: float # a score of how important this action is, from 0.0 to 1.0
    # TODO: make this priority field more robust

class GenericActionBase(ActionBase):
    """
    Generic app-agnostic action.
    """
    package: Optional[str] = None

class InstagramActionBase(ActionBase):
    """
    Instagram-specific action base.
    TODO: expand fields.
    """
    package: str = "com.instagram.android"
    username: str

class WhatsAppActionBase(ActionBase):
    """
    WhatsApp-specific action base.
    TODO: expand fields.
    """
    package: str = "com.whatsapp"
    recipient_name: str

class SlackActionBase(ActionBase):
    """
    Slack-specific action base.
    TODO: expand fields.
    """
    package: str = "com.slack"
    channel_name: str
    thread_name: str
    recipient_name: str
