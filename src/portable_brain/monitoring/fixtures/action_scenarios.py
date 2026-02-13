# pre-defined action scenarios to allow replays and mock testing
# NOTE: each scenario returns a list[Action] that can be fed directly into replay_action_sequence() inside ObservationTracker.
# Timestamps are deterministic (fixed Jan 2025 dates) so scenarios are repeatable.

from datetime import datetime
from typing import Callable

from portable_brain.monitoring.background_tasks.types.action.actions import (
    Action,
    AppSwitchAction,
    InstagramMessageSentAction,
    InstagramPostLikedAction,
    WhatsAppMessageSentAction,
    SlackMessageSentAction,
)
from portable_brain.monitoring.background_tasks.types.ui_states.state_changes import StateChangeSource
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType

# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def _t(day: int, hour: int, minute: int = 0) -> datetime:
    """Build a deterministic fixture timestamp (Jan 2025)."""
    return datetime(2025, 1, day, hour, minute, 0)

_SOURCE = StateChangeSource.OBSERVATION # all fixture actions are treated as observed

# ---------------------------------------------------------------------------
# Scenario 1: Frequent Instagram DMs to a close contact
# Expected observation -> long_term_people (sarah_smith, Instagram as primary channel)
# ---------------------------------------------------------------------------

def instagram_close_friend_messaging() -> list[Action]:
    """
    12 Instagram DMs to sarah_smith across 3 days at varying times.
    Exercises: InstagramMessageSentAction, recurring same-target communication.
    """
    return [
        # Day 1 — Jan 15
        InstagramMessageSentAction(
            timestamp=_t(15, 9, 23), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        InstagramMessageSentAction(
            timestamp=_t(15, 14, 45), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        InstagramMessageSentAction(
            timestamp=_t(15, 20, 12), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        # Day 2 — Jan 16
        InstagramMessageSentAction(
            timestamp=_t(16, 10, 5), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        InstagramMessageSentAction(
            timestamp=_t(16, 13, 30), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        InstagramMessageSentAction(
            timestamp=_t(16, 21, 0), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        # Day 3 — Jan 17
        InstagramMessageSentAction(
            timestamp=_t(17, 9, 50), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        InstagramMessageSentAction(
            timestamp=_t(17, 15, 10), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
        InstagramMessageSentAction(
            timestamp=_t(17, 19, 40), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            actor_username="john_doe", target_username="sarah_smith",
        ),
    ]

# ---------------------------------------------------------------------------
# Scenario 2: Morning work app routine (Gmail -> Slack)
# Expected observation -> long_term_preferences (established morning workflow)
# ---------------------------------------------------------------------------

def morning_work_app_routine() -> list[Action]:
    """
    Launcher→Gmail→Slack sequence repeated across 3 days at ~9 AM.
    Exercises: AppSwitchAction, sequential temporal routine.
    """
    return [
        # Day 1 — Jan 15
        AppSwitchAction(
            timestamp=_t(15, 9, 0), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.google.android.gm",
            src_package="com.android.launcher", dst_package="com.google.android.gm",
        ),
        AppSwitchAction(
            timestamp=_t(15, 9, 5), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.slack",
            src_package="com.google.android.gm", dst_package="com.slack",
        ),
        # Day 2 — Jan 16
        AppSwitchAction(
            timestamp=_t(16, 9, 2), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.google.android.gm",
            src_package="com.android.launcher", dst_package="com.google.android.gm",
        ),
        AppSwitchAction(
            timestamp=_t(16, 9, 6), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.slack",
            src_package="com.google.android.gm", dst_package="com.slack",
        ),
        # Day 3 — Jan 17
        AppSwitchAction(
            timestamp=_t(17, 9, 1), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.google.android.gm",
            src_package="com.android.launcher", dst_package="com.google.android.gm",
        ),
        AppSwitchAction(
            timestamp=_t(17, 9, 6), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.slack",
            src_package="com.google.android.gm", dst_package="com.slack",
        ),
    ]

# ---------------------------------------------------------------------------
# Scenario 3: Cross-platform contact communication (work vs. personal)
# Expected observation -> long_term_people (mike_johnson, platform separation by context)
# ---------------------------------------------------------------------------

def cross_platform_contact_communication() -> list[Action]:
    """
    Slack DMs to mike_johnson during work hours, then WhatsApp after 6 PM.
    Exercises: SlackMessageSentAction, WhatsAppMessageSentAction, AppSwitchAction, platform separation.
    """
    return [
        # Day 1 — Jan 15: work hours on Slack
        SlackMessageSentAction(
            timestamp=_t(15, 10, 30), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            workspace_name="TechCorp", channel_name="engineering", thread_name=None,
            target_name="mike_johnson", is_dm=True,
        ),
        SlackMessageSentAction(
            timestamp=_t(15, 14, 20), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            workspace_name="TechCorp", channel_name="engineering", thread_name=None,
            target_name="mike_johnson", is_dm=True,
        ),
        SlackMessageSentAction(
            timestamp=_t(15, 16, 45), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            workspace_name="TechCorp", channel_name="engineering", thread_name=None,
            target_name="mike_johnson", is_dm=True,
        ),
        # Day 1 — Jan 15: after-work switch to WhatsApp
        AppSwitchAction(
            timestamp=_t(15, 18, 5), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.whatsapp",
            src_package="com.slack", dst_package="com.whatsapp",
        ),
        WhatsAppMessageSentAction(
            timestamp=_t(15, 18, 6), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            recipient_name="mike_johnson", target_name="mike_johnson", is_dm=True,
        ),
        # Day 2 — Jan 16: same pattern
        SlackMessageSentAction(
            timestamp=_t(16, 11, 0), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            workspace_name="TechCorp", channel_name="engineering", thread_name=None,
            target_name="mike_johnson", is_dm=True,
        ),
        SlackMessageSentAction(
            timestamp=_t(16, 15, 30), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            workspace_name="TechCorp", channel_name="engineering", thread_name=None,
            target_name="mike_johnson", is_dm=True,
        ),
        AppSwitchAction(
            timestamp=_t(16, 18, 10), source=_SOURCE, source_change_type=StateChangeType.APP_SWITCH,
            package="com.whatsapp",
            src_package="com.slack", dst_package="com.whatsapp",
        ),
        WhatsAppMessageSentAction(
            timestamp=_t(16, 18, 11), source=_SOURCE, source_change_type=StateChangeType.TEXT_INPUT,
            recipient_name="mike_johnson", target_name="mike_johnson", is_dm=True,
        ),
    ]

# ---------------------------------------------------------------------------
# Scenario 4: Instagram fitness content browsing
# Expected observation -> long_term_preferences (recurring evening fitness content engagement)
# ---------------------------------------------------------------------------

def instagram_fitness_content_browsing() -> list[Action]:
    """
    Repeated likes on fitness accounts (fitness_coach_alex, healthy_recipes_daily) in the evening.
    Exercises: InstagramPostLikedAction, content engagement pattern, temporal clustering.
    """
    return [
        # Day 1 — Jan 15 (evening)
        InstagramPostLikedAction(
            timestamp=_t(15, 19, 15), source=_SOURCE, source_change_type=StateChangeType.CONTENT_NAVIGATION,
            actor_username="john_doe", target_username="fitness_coach_alex",
        ),
        InstagramPostLikedAction(
            timestamp=_t(15, 19, 18), source=_SOURCE, source_change_type=StateChangeType.CONTENT_NAVIGATION,
            actor_username="john_doe", target_username="fitness_coach_alex",
        ),
        # Day 2 — Jan 16 (evening)
        InstagramPostLikedAction(
            timestamp=_t(16, 20, 22), source=_SOURCE, source_change_type=StateChangeType.CONTENT_NAVIGATION,
            actor_username="john_doe", target_username="fitness_coach_alex",
        ),
        InstagramPostLikedAction(
            timestamp=_t(16, 20, 25), source=_SOURCE, source_change_type=StateChangeType.CONTENT_NAVIGATION,
            actor_username="john_doe", target_username="healthy_recipes_daily",
        ),
        # Day 3 — Jan 17 (evening)
        InstagramPostLikedAction(
            timestamp=_t(17, 19, 10), source=_SOURCE, source_change_type=StateChangeType.CONTENT_NAVIGATION,
            actor_username="john_doe", target_username="fitness_coach_alex",
        ),
        InstagramPostLikedAction(
            timestamp=_t(17, 19, 14), source=_SOURCE, source_change_type=StateChangeType.CONTENT_NAVIGATION,
            actor_username="john_doe", target_username="fitness_coach_alex",
        ),
    ]

# ---------------------------------------------------------------------------
# Scenario registry
# Maps scenario name (used by the replay route) -> factory function.
# Add new scenarios here as they are defined above.
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Callable[[], list[Action]]] = {
    "instagram_close_friend_messaging": instagram_close_friend_messaging,
    "morning_work_app_routine": morning_work_app_routine,
    "cross_platform_contact_communication": cross_platform_contact_communication,
    "instagram_fitness_content_browsing": instagram_fitness_content_browsing,
}
