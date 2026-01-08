from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from portable_brain.monitoring.background_tasks.types.ui_states.state_changes import StateChangeSource
from datetime import datetime, timezone, timedelta

class ActionBase(BaseModel):
    """
    High-level inference for what "action" was done by the user.
    - Inferred based on low-level UI change signals.

    NOTE: This is the absolute base class for all actions and shared metadata.
    - App-specific action bases inherit from this base with specific protocols.
    """
    timestamp: datetime
    description: Optional[str] = None # Human-readable description
    source: StateChangeSource # propagated from changes in UI state
    importance: float # a score of how important this action is, from 0.0 to 1.0
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
    actor_username: str # the device user's current account username

class WhatsAppActionBase(ActionBase):
    """
    WhatsApp-specific action base.
    TODO: expand fields.
    """
    package: str = "com.whatsapp"
    recipient_name: str
    is_dm: bool

class SlackActionBase(ActionBase):
    """
    Slack-specific action base.
    TODO: expand fields.
    """
    package: str = "com.slack"
    workspace_name: str
    channel_name: str
    thread_name: Optional[str]
    is_dm: bool
