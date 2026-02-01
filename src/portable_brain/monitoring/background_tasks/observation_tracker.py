# src/monitoring/observation_tracker.py
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.common.types.android_apps import AndroidApp
from portable_brain.common.logging.logger import logger

# Canonical DTOs for UI state, inferred action, observations
from portable_brain.monitoring.background_tasks.types.ui_states.ui_state import UIState, UIActivity
from portable_brain.monitoring.background_tasks.types.ui_states.state_changes import UIStateChange, StateChangeSource
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType
from portable_brain.monitoring.background_tasks.types.action.action_types import ActionType
from portable_brain.monitoring.background_tasks.types.action.actions import (
    Action,
    AppSwitchAction,
    UnknownAction,
    InstagramMessageSentAction,
    InstagramPostLikedAction,
    WhatsAppMessageSentAction,
    SlackMessageSentAction,
    # TBD
)
from portable_brain.monitoring.background_tasks.types.observation.observations import (
    Observation,
    LongTermPeopleObservation,
    LongTermPreferencesObservation,
    ShortTermPreferencesObservation,
    ShortTermContentObservation
)

# LLM for inference
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient

from collections import deque

class ObservationTracker:
    """
    Track ALL device state changes, including manual user actions.
    - Client refers to the main DroidRunClient instance.
    - The client's helper are used to detect state changes.

    Complements DroidRunClient.action_history which only tracks agent-executed actions via execute_command()

    TODO: finish implementing this tracker.
    - Also create canonical DTO for observations and enums for actions
    """

    def __init__(self, droidrun_client: DroidRunClient):
        self.droidrun_client = droidrun_client
        # track the 50 most recent inferred actions
        self.inferred_actions: deque[Action] = deque(maxlen=50)
        # track the 20 most recent high-level observations based on inferred actions
        # NOTE: observations look at prev. records to update
        self.observations: deque[Observation] = deque(maxlen=20)
        # store recent state changes as a queue w/ max length of 10 to avoid too much memory
        self.recent_state_changes: deque[UIStateChange] = deque(maxlen=10)
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
                change: UIStateChange | None = await self.droidrun_client.detect_state_change()

                if change:
                    # track of the most recent state changes
                    # NOTE: automatically maintained via deque
                    self.recent_state_changes.append(change)
                    logger.info(f"Detected state change: {change.change_type}")
                    # infer what action might have caused this change
                    inferred_action = self._infer_action(change)
                    # store inferred actions
                    self.inferred_actions.append(inferred_action)

                    # TODO: should be handled by memory handler in future
                    # await self.memory_handler.process_observation(observation)
                    # shorter cooldown if state change HAS been found -> likely another action might pursue
                    await asyncio.sleep(0.2)
                else:
                    await asyncio.sleep(poll_interval) # cooldown after each iteration

            except Exception as e:
                print(f"Observation tracking error: {e}")
                await asyncio.sleep(5)  # Back off on error

    def _infer_action(self, change: UIStateChange) -> Action:
        """
        Infers a user action from recorded UI state change via rule-based classification.
        Returns an Action object, based on change type and state metadata.
        Returns UnknownAction if action can't be inferred.
        """

        # parse change
        change_type: StateChangeType = change.change_type
        before: UIState = change.before # states
        after: UIState = change.after # states
        curr_package: str = before.package # the current app is treated as the package BEFORE change

        # TODO: infer all actions based on each change type and state metadata
        if change.change_type == StateChangeType.APP_SWITCH:
            logger.info(f"Inferred app switch action from change type: {change_type} and package: {before.package} to {after.package}")
            return AppSwitchAction(
                timestamp=change.timestamp,
                source_change_type=change.change_type,
                package=change.after.package,
                source=change.source,
                description=change.description,
                src_package=change.before.package,
                src_activity=change.before.activity,
                dst_package=change.after.package,
                dst_activity=change.after.activity,
            )

        # else, see if current app is supported
        elif curr_package == AndroidApp.INSTAGRAM:
            if change_type == StateChangeType.TEXT_INPUT:
                logger.info(f"Inferred Instagram message sent action from change type: {change_type} and package: {curr_package}")
                return InstagramMessageSentAction(
                    timestamp=change.timestamp,
                    source_change_type=change.change_type,
                    actor_username=change.after.raw_tree.get("username", "unknown user") if change.after.raw_tree else "unknown user", # actor username
                    target_username=change.after.raw_tree.get("target_username", "unknown user") if change.after.raw_tree else "unknown user", # target username
                    source=change.source,
                    # importance is set to default 1.0 for now
                    description=change.description,
                    message_summary=None, # should use UI states diff to infer message summary w/ LLM
                )
            # otherwise no other actions are supported for Instagram, so return unknown

        elif curr_package == AndroidApp.WHATSAPP:
            if change_type == StateChangeType.TEXT_INPUT:
                logger.info(f"inferred WhatsApp message sent action from change type: {change_type} and package: {curr_package}")
                return WhatsAppMessageSentAction(
                    timestamp=change.timestamp,
                    source_change_type=change.change_type,
                    recipient_name=change.after.raw_tree.get("recipient_name", "unknown user") if change.after.raw_tree else "unknown user",
                    is_dm=change.after.raw_tree.get("is_dm", False) if change.after.raw_tree else False,
                    target_name=change.after.raw_tree.get("target_name", "unknown user") if change.after.raw_tree else "unknown user",
                    source=change.source,
                    # importance is set to default 1.0 for now
                    description=change.description,
                    message_summary=None,
                )
            # otherwise no other actions are supported for WhatsApp, so return unknown
        
        elif curr_package == AndroidApp.SLACK:
            if change_type == StateChangeType.TEXT_INPUT:
                logger.info(f"Inferred Slack message sent action from change type: {change_type} and package: {curr_package}")
                return SlackMessageSentAction(
                    timestamp=change.timestamp,
                    source_change_type=change.change_type,
                    workspace_name=change.after.raw_tree.get("workspace_name", "unknown workspace") if change.after.raw_tree else "unknown workspace",
                    channel_name=change.after.raw_tree.get("channel_name", "unknown channel") if change.after.raw_tree else "unknown channel",
                    thread_name=change.after.raw_tree.get("thread_name", None) if change.after.raw_tree else None,
                    target_name=change.after.raw_tree.get("target_name", "unknown user") if change.after.raw_tree else "unknown user",
                    source=change.source,
                    # importance is set to default 1.0 for now
                    description=change.description,
                    message_summary=None,
                )
            # otherwise no other actions are supported for Slack, so return unknown

        # if action can't be inferred, return UnknownAction 
        logger.info(f"Unable to infer action from change type: {change_type} and package: {curr_package}")
        return UnknownAction(
            timestamp=change.timestamp,
            source_change_type=change.change_type,
            package=change.after.package,
            source=change.source,
            importance=0.0, # TEMP: for unknown actions, we override importance to 0.0
            description=change.description,
        )

    def _create_observation(self, context_size: int = 10) -> Optional[Observation]:
        """
        Creates a final observation object based on the current history of actions.
            - An observation object will be one of the possible memory nodes.
        This is a high-level abstraction derived from a union of low-level actions.
        NOTE: observation is what's ultimately stored in the memory.
        - Returns None if no meaningful observation can be made.

        Observations are made in a continuous, sequential environment, so to prevent duplicates:
            - Utilize an appropriate window size of actions (context_size)
                - This helper should only be called every context_size new actions are recorded
            - Look at previous time step's observation, and either update it based on new context or create a new observation
        """
        # TODO: use the history of inferred actions to build more observations
        # for now, just build a single observation to test

        # if no inferred actions, return; there are no observations to make
        if not self.inferred_actions:
            return None
        
        recent_actions = list(self.inferred_actions)[-context_size:]
        last_observation = self.observations[-1] if self.observations else None

        # create new observation or update previous
        new_observation: Observation | None = None
        if last_observation:
            # if we have a recent observation, either update it or create new
            # compare previous observation in the context of recent actions
            
            # logic for looking at recent actions and making a meaningful observation
            pass
        else:
            # otherwise, create a new observation unconditionally
            pass

        # TODO: load in llm client and use semantic parsing
        # short -> long term storage is only relevant for preferences
            
        # return new observation (may be None)
        return new_observation

    def get_inferred_actions(
        self,
        limit: Optional[int] = None,
        change_types: Optional[list[StateChangeType]] = None,
    ) -> List[Action]:
        """
        Get inferred actions history.
        NOTE: only upto 50 recent actions are stored

        Args:
            limit: Max observations to return
            change_types: Filter by change type enums.

        Returns:
            List of observations
            NOTE: the bottom index in returned list is the most recent, so return is reversed.
        """
        # wrap in list to allow negative idx slicing
        inferred_actions = list(self.inferred_actions)

        # optional filtering by change type
        if change_types:
            inferred_actions = [
                action for action in inferred_actions
                if action.source_change_type in set(change_types)
            ]

        # optional filtering by number of actions limit
        if limit:
            inferred_actions = inferred_actions[-limit:] # takes the last limit number of inferred actions
        
        inferred_actions.reverse() # make the first observation the most recent

        return inferred_actions

    def get_observations(
        self,
        limit: Optional[int] = None,
    ) -> List: # need observation DTO
        """
        Get observation history.
        TODO: need an observation DTO.

        Args:
            limit: Max observations to return

        Returns:
            List of observations
            NOTE: the bottom index in returned list is the most recent. Possibly reverse indices to fetch most recent on top.
        """
        observations = list(self.observations)

        # optional filtering by number of observations limit
        if limit:
            observations = observations[-limit:]
        
        observations.reverse() # make the first observation the most recent

        return observations
    
    def get_state_changes(
        self,
        limit: Optional[int] = None, # NOTE: tracker only stores the last 10 state changes right now
        change_types: Optional[list[StateChangeType]] = None,
    ) -> List[UIStateChange]:
        """
        Get state change history.

        Args:
            limit: Max state changes to return (only the last 10 are stored anyway)
            change_types: Filter by change type enums

        Returns:
            List of recent state changes
            NOTE: the bottom index in returned list is the most recent. Possibly reverse indices to fetch most recent on top.
        """
        # wrap state changes in a list to allow negative idx slicing for limit
        state_changes = list(self.recent_state_changes)

        # optional filtering by change type
        if change_types:
            state_changes = [
                change for change in state_changes
                if change.change_type in set(change_types)
            ]
        
        # optional filtering by number of observations limit
        if limit:
            state_changes = state_changes[-limit:]
        
        state_changes.reverse() # make the first observation the most recent

        return state_changes

    def clear_observations(self):
        """Clear observation history after persisting to DB."""
        self.observations.clear()

    def clear_inferred_actions(self):
        """Clear inferred action history after persisting to DB."""
        self.inferred_actions.clear()

    def clear_state_changes(self):
        """Clear state change history after persisting to DB."""
        self.recent_state_changes.clear()

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
    