# The tracker for monitoring low-level HCI data as a background task
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.common.types.android_apps import AndroidApp
from portable_brain.common.logging.logger import logger

# Base repository for all observations
from portable_brain.monitoring.observation_repository import ObservationRepository

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
# helper to persist memory in structured db
from portable_brain.common.db.crud.memory.structured_memory_crud import save_observation_to_structured_memory
# helper to persis memory in text log (vector db) NOTE: used implicitly by generator client right now
from portable_brain.common.db.crud.memory.text_embeddings_crud import save_text_embedding_log

# LLM for inference
from portable_brain.common.services.llm_service.llm_client import TypedLLMClient
# helper class to infer observations
from portable_brain.monitoring.semantic_filtering.llm_filtering.observations import ObservationInferencer

# Text embedding client for generation
from portable_brain.common.services.embedding_service.text_embedding import TypedTextEmbeddingClient
# helper class to generate embeddings
from portable_brain.monitoring.embedding_manager.text_embeddings.generate_embeddings import EmbeddingGenerator

# data structrue to track only recent information
from collections import deque
# async engine for db
from sqlalchemy.ext.asyncio import AsyncEngine
from portable_brain.common.db.session import get_async_session_maker

class ObservationTracker(ObservationRepository):
    """
    Track ALL device state changes, including manual user actions.
    - Client refers to the main DroidRunClient instance.
    - The client's helper are used to detect state changes.

    Complements DroidRunClient.action_history which only tracks agent-executed actions via execute_command()

    TODO: finish implementing this tracker.
    - Also create canonical DTO for observations and enums for actions
    """

    def __init__(self, droidrun_client: DroidRunClient, llm_client: TypedLLMClient, text_embedding_client: TypedTextEmbeddingClient, main_db_engine: AsyncEngine):
        # NOTE: if tracker holds any additional dependencies in the future, the items from repository needs to be re-initialized.
        super().__init__(droidrun_client=droidrun_client, llm_client=llm_client, main_db_engine=main_db_engine)
        # track the 50 most recent inferred actions
        self.inferred_actions: deque[Action] = deque(maxlen=50)
        # NOTE: should be 10, lowered for testing
        self.action_context_size: int = 3 # number of previous actions to track before attempting to infer an observation
        self.action_counter: int = 0 # counter to track the number of actions since last observation
        # track the 20 most recent high-level observations based on inferred actions
        # NOTE: observations look at prev. records to update
        self.observations: deque[Observation] = deque(maxlen=20)
        # store recent state changes as a queue w/ max length of 10 to avoid too much memory
        self.recent_state_changes: deque[UIStateChange] = deque(maxlen=10)
        self.running = False
        self._tracking_task: Optional[asyncio.Task] = None
        # observation helper
        self.inferencer = ObservationInferencer(droidrun_client=self.droidrun_client, llm_client=self.llm_client, main_db_engine=self.main_db_engine)
        # embedding helper NOTE: embedding client is not a core dependency of observation tracker.
        self.embedding_generator = EmbeddingGenerator(embedding_client=text_embedding_client, main_db_engine=self.main_db_engine)

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

                    self.action_counter += 1
                    # penultimate step, create observation node every context_size actions
                    if self.action_counter >= self.action_context_size:
                        new_observation = await self._create_or_update_observation(context_size=self.action_context_size)
                        if new_observation:
                            # save observation to memory db and local history
                            # TODO: this helper should evict the old observation, save that to db, and add new observation to local history
                            await self._save_observation(new_observation)
                        # reset counter
                        self.action_counter = 0

                    # TODO: final step, should be handled by memory handler in future
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

    async def _create_or_update_observation(self, context_size: int = 10) -> Optional[Observation]:
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

        NOTE: if a previous observation should be updated, handles local history and returns None
        """
        # if no inferred actions, return; there are no observations to make or update
        if not self.inferred_actions:
            return None
        
        recent_actions = list(self.inferred_actions)[-context_size:]
        last_observation = self.observations[-1] if self.observations else None

        # create new observation or update previous
        new_observation: Observation | None = None
        updated_observation: Observation | None = None # conditionally update
        if last_observation:
            # if we have a recent observation, either update it or create new
            # compare previous observation in the context of recent actions, try to update it
            updated_observation = await self.inferencer.update_observation(last_observation, recent_actions)
        if not updated_observation:
            # if we reach here, then either no last observation, or there is nothing meaningful to update
            # -> create new observation; look at recent actions and make a meaningful observation
            logger.info(f"No update to make, creating new observation from recent actions.")
            new_observation = await self.inferencer.create_new_observation(recent_actions)
            logger.info(f"Created new observation from recent actions: {new_observation.node if new_observation else None}")
            # NOTE: the parent caller will handle saving the cache-evicted observation to db
            return new_observation # may be None, which indicates no new, meaningful observation to make
        else:
            # there is a meaningful observation to update, so update local history and return None
            logger.info(f"Updated observation from recent actions: {updated_observation.node}")
            self.observations[-1] = updated_observation # replace last observation w/ updated
            return None
        
        # TODO: load in existing nodes by semantic similarity and update or make edges
        # short -> long term storage is only relevant for preferences
    
    async def _save_observation(self, new_observation: Observation) -> None:
        """
        Save observation to memory db and local history.
        NOTE: this is the baseline database for now.
        """

        # evict old observation from local history
        # NOTE: we're not truly evicting yet (max history is 20, for debugging), but we update the current last observation to DB
        old_observation = self.observations[-1]
        if old_observation:
            # let helper save old observation to structured memory

            # NOTE: temporarily disabled, until textlog completed and clearer memory structure is defined.
            # await save_observation_to_structured_memory(old_observation, self.main_db_engine)
            # logger.info(f"Successfully saved old observation to STRUCTURED MEMORY: {old_observation.node}")
            
            # also saves to text log (semantic vector db) NOTE: this logic might be temporary.
            # we also use a convenience wrapper that handles both embedding generation and saving; should separate in future.
            await self.embedding_generator.generate_and_save_embedding(observation_id=old_observation.id, observation_text=old_observation.node)
            logger.info(f"Successfully saved old observation to TEXT LOG: {old_observation.node}")
        # saves new observation to local history
        self.observations.append(new_observation)
            
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

    # helper to monitor states of tracker
    def get_monitoring_overview(self):
        """
        Lightweight helper to get overview of monitoring history.
        Currently supports fetching the length of each history.
        """
        return {
            "inferred_actions": len(self.inferred_actions),
            "observations": len(self.observations),
            "state_changes": len(self.recent_state_changes),
        }

    def start_background_tracking(self, poll_interval: float = 1.0):
        """
        Start tracking as a background task.
        Currently called by the background tracking endpoint.

        Args:
            poll_interval: How often to poll for changes (seconds)
        """
        if self._tracking_task is not None and self._tracking_task.done():
            self._tracking_task = None

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

        # Clear the task reference after cleanup
        self._tracking_task = None
    
    async def create_test_observation(self, context_size: int = 10) -> Optional[Observation]:
        """
        Creates a TEST observation object based on the current history of actions.
        Verifies LLM / RAG functionality.
        """
        # TODO: use the history of inferred actions to build more observations
        # for now, just build a single observation to test

        # if no inferred actions, return; there are no observations to make
        if not self.inferred_actions:
            logger.info("no inferred actions to create observation from")
            return None
        
        recent_actions = list(self.inferred_actions)[-context_size:]
        last_observation = self.observations[-1] if self.observations else None

        # NOTE: test this after verifying creation works
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

        # for now, unconditional test
        # use helper to create observation
        new_observation = await self.inferencer.test_create_new_observation(actions=recent_actions)

        # TODO: load in llm client and use semantic parsing
        # short -> long term storage is only relevant for preferences
            
        # return new observation (may be None)
        return new_observation
