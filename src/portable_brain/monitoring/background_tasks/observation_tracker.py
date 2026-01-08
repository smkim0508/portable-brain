# src/monitoring/observation_tracker.py
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.monitoring.background_tasks.types.action.actions import (
    Action,
    AppSwitchAction,
    InstagramMessageSentAction,
    InstagramPostLikedAction,
    WhatsAppMessageSentAction,
    SlackMessageSentAction,
    # TBD
)
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType
from portable_brain.common.types.android_apps import AndroidApp
from portable_brain.common.logging.logger import logger

class ObservationTracker:
    """
    Track ALL device state changes, including manual user actions.

    This complements DroidRunClient.action_history which only tracks
    execute_command() actions.

    TODO: finish implementing this tracker.
    - Also create canonical DTO for observations and enums for actions
    """

    def __init__(self, client: DroidRunClient):
        self.client = client
        self.observations: List[Dict[str, Any]] = []
        self.running = False
        self._tracking_task: Optional[asyncio.Task] = None

    async def start_tracking(self, poll_interval: float = 1.0):
        """
        Start continuous observation tracking.

        Args:
            poll_interval: How often to poll for changes (seconds)
        """
        self.running = True

        while self.running:
            try:
                # Detect any state change
                change = await self.client.detect_state_change()

                if change:
                    # Infer what action might have caused this change
                    observation = self._create_observation(change)

                    # Store observation
                    self.observations.append(observation)

                    # Optional: Send to memory handler immediately
                    # await self.memory_handler.process_observation(observation)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                print(f"Observation tracking error: {e}")
                await asyncio.sleep(5)  # Back off on error

    def _create_observation(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create observation record from state change.

        Infers what action likely occurred based on the change type.
        """
        observation = {
            "timestamp": datetime.now().isoformat(),
            "change_type": change["change_type"],
            "before": change["before"],
            "after": change["after"],
            "inferred_action": self._infer_action(change),
            "source": "observation",  # vs "command" from execute_command
        }

        return observation

    def _infer_action(self, change: Dict[str, Any]) -> Optional[Action]:
        """
        Infer what user action likely caused this state change.
        Returns an Action object or None if action is determined to be unknown.
        TODO: add more sophisticated logic.
        """
        change_type: StateChangeType = change["change_type"] # TODO: add change DTO
        before = change["before"] # states
        after = change["after"] # states
        curr_package = before["package"]

        if change_type == StateChangeType.APP_SWITCH:
            return AppSwitchAction(
                timestamp=change["timestamp"],
                package=after["package"],
                source=change["source"],
                priority=change["priority"],
                description=change["description"],
                src_package=before["package"],
                src_activity=before["activity"],
                dst_package=after["package"],
                dst_activity=after["activity"],
            )

        # else, see if current app supports special tracking
        elif curr_package == AndroidApp.INSTAGRAM:
            if change_type == StateChangeType.TEXT_INPUT:
               return InstagramMessageSentAction(
                   timestamp=change["timestamp"],
                   actor_username=change["username"], # actor username
                   target_username=change["target_username"],
                   source=change["source"],
                   priority=change["priority"],
                   description=change["description"],
                   message_summary=change["message_summary"],
               )
            else:
                return None

        elif curr_package == AndroidApp.WHATSAPP:
            if change_type == StateChangeType.TEXT_INPUT:
               return WhatsAppMessageSentAction(
                   timestamp=change["timestamp"],
                   recipient_name=change["name"],
                   is_dm=change["is_dm"],
                   target_name=change["target_name"],
                   source=change["source"],
                   priority=change["priority"],
                   description=change["description"],
                   message_summary=change["message_summary"],
               )
            else:
                return None
        
        elif curr_package == AndroidApp.SLACK:
            if change_type == StateChangeType.TEXT_INPUT:
               return SlackMessageSentAction(
                   timestamp=change["timestamp"],
                   workspace_name=change["workspace_name"],
                   channel_name=change["channel_name"],
                   thread_name=change["thread_name"],
                   target_name=change["target_name"],
                   source=change["source"],
                   priority=change["priority"],
                   description=change["description"],
                   message_summary=change["message_summary"],
               )
            else:
                return None
        else:
            logger.info(f"Unknown action, change type: {change_type}")
            return None

    def get_observations(
        self,
        limit: Optional[int] = None,
        change_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get observation history.

        Args:
            limit: Max observations to return
            change_types: Filter by change types (e.g., ['app_switch', 'screen_change'])

        Returns:
            List of observations
            NOTE: the bottom index in returned list is the most recent. Possibly reverse indices to fetch most recent on top.
        """
        observations = self.observations

        if change_types:
            observations = [
                o for o in observations
                if o["change_type"] in change_types
            ]

        if limit:
            observations = observations[-limit:]
        
        observations.reverse() # make the first observation the most recent

        return observations

    def clear_observations(self):
        """Clear observation history after persisting to DB."""
        self.observations = []

    def start_background_tracking(self, poll_interval: float = 1.0):
        """
        Start tracking as a background task.
        Currently called by the background tracking endpoint.

        Args:
            poll_interval: How often to poll for changes (seconds)
        """
        if self._tracking_task is not None and not self._tracking_task.done():
            raise RuntimeError("Observation tracking already running")

        self._tracking_task = asyncio.create_task(self.start_tracking(poll_interval))
        return self._tracking_task

    async def stop_tracking(self):
        """
        Stop observation tracking and wait for cleanup.
        Call this in lifespan shutdown.
        """
        self.running = False

        # Wait for the tracking loop to exit gracefully
        if self._tracking_task is not None and not self._tracking_task.done():
            try:
                # Give it a moment to finish current iteration
                await asyncio.wait_for(self._tracking_task, timeout=5.0)
            except asyncio.TimeoutError:
                # Force cancel if it doesn't stop gracefully
                self._tracking_task.cancel()
                try:
                    await self._tracking_task
                except asyncio.CancelledError:
                    pass  # Expected
    