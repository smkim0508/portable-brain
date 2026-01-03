# src/monitoring/observation_tracker.py
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient

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

    def _infer_action(self, change: Dict[str, Any]) -> str:
        """
        Infer what user action likely caused this state change.
        """
        change_type = change["change_type"]
        before = change["before"]
        after = change["after"]

        if change_type == "app_switch":
            return f"switched_from_{before['package']}_to_{after['package']}"

        elif change_type == "screen_change":
            return f"navigated_to_{after['activity']}"

        elif change_type == "major_layout_change":
            # Could be: dialog opened, form submitted, content loaded
            if after["element_count"] > before["element_count"]:
                return "content_expanded"  # e.g., dropdown opened
            else:
                return "content_collapsed"  # e.g., dialog closed

        elif change_type == "minor_layout_change":
            # Could be: text input, selection change
            return "ui_interaction"

        else:
            return "unknown_action"

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
        """
        observations = self.observations

        if change_types:
            observations = [
                o for o in observations
                if o["change_type"] in change_types
            ]

        if limit:
            observations = observations[-limit:]

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
    