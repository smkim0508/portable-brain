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

# Canonical DTOs for UI state, state snapshots, observations
from portable_brain.monitoring.background_tasks.types.ui_states.ui_state import UIState, UIActivity
from portable_brain.monitoring.background_tasks.types.ui_states.state_changes import UIStateChange, StateChangeSource
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType
from portable_brain.monitoring.background_tasks.types.ui_states.state_snapshot import UIStateSnapshot

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

    Complements DroidRunClient.execution_history which only tracks agent-executed actions via execute_command()

    TODO: finish implementing this tracker.
    """

    def __init__(self, droidrun_client: DroidRunClient, llm_client: TypedLLMClient, text_embedding_client: TypedTextEmbeddingClient, main_db_engine: AsyncEngine):
        # NOTE: if tracker holds any additional dependencies in the future, the items from repository needs to be re-initialized.
        super().__init__(droidrun_client=droidrun_client, llm_client=llm_client, main_db_engine=main_db_engine)
        # tracker settings
        self.last_poll_interval: float = 1.0 # saves the last polling interval to preserve it after pauses
        self.snapshot_context_size: int = 10

        # tracker states
        self.running = False
        self._tracking_task: Optional[asyncio.Task] = None
        self.snapshot_counter: int = 0

        # track the 50 most recent state snapshots as structured DTOs
        self.state_snapshots: deque[UIStateSnapshot] = deque(maxlen=50)

        # track the 20 most recent high-level observations based on semantic state snapshots
        # NOTE: observations look at prev. records to update
        self.observations: deque[Observation] = deque(maxlen=20)

        # store recent state changes as a queue w/ max length of 10 to avoid too much memory
        self.recent_state_changes: deque[UIStateChange] = deque(maxlen=10)

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
                # returns None if no change, otherwise a UIStateChange object
                change: UIStateChange | None = await self.droidrun_client.detect_state_change()

                if change:
                    # track of the most recent state changes
                    # NOTE: automatically maintained via deque
                    self.recent_state_changes.append(change)
                    logger.info(f"Detected state change: {change.change_type}")
                    
                    # construct a UIStateSnapshot DTO from the detected change
                    is_app_switch = change.change_type == StateChangeType.APP_SWITCH
                    snapshot = UIStateSnapshot(
                        formatted_text=change.after.formatted_text,
                        activity=change.after.activity,
                        package=change.after.package,
                        timestamp=change.timestamp,
                        is_app_switch=is_app_switch,
                        app_switch_info=f"APP SWITCH: from {change.before.package} to {change.after.package}" if is_app_switch else None,
                    )
                    self.state_snapshots.append(snapshot)

                    # no more fragile inference on actions

                    self.snapshot_counter += 1
                    # penultimate step, create observation node every context_size snapshots
                    if self.snapshot_counter >= self.snapshot_context_size:
                        new_observation = await self._create_or_update_observation(context_size=self.snapshot_context_size)
                        if new_observation:
                            # save observation to memory db and local history
                            # TODO: this helper should evict the old observation, save that to db, and add new observation to local history
                            await self._save_observation(new_observation)

                            # TODO: final step, should be handled by memory handler in future
                            # NOTE: this step is to ensure new observations are processed in memory to do temporal update / rearranging
                            # await self.memory_handler.process_observation(observation)

                        # reset counter
                        self.snapshot_counter = 0

                    # shorter cooldown if state change HAS been found -> likely another state change might pursue
                    await asyncio.sleep(0.2)
                else:
                    await asyncio.sleep(poll_interval) # cooldown after each iteration

            except Exception as e:
                print(f"Observation tracking error: {e}")
                await asyncio.sleep(5) # Back off on error

    async def _create_or_update_observation(self, context_size: int = 10) -> Optional[Observation]:
        """
        Creates a final observation object based on the current history of state snapshots.
            - An observation object will be one of the possible memory nodes.
        This is a high-level abstraction derived from a union of low-level UI snapshots.
        NOTE: observation is what's ultimately stored in the memory.
        - Returns None if no meaningful observation can be made.

        Observations are made in a continuous, sequential environment, so to prevent duplicates:
            - Utilize an appropriate window size of snapshots (context_size)
                - This helper should only be called every context_size new snapshots are recorded
            - Look at previous time step's observation, and either update it based on new context or create a new observation

        NOTE: if a previous observation should be updated, handles local history and returns None
        """
        # if no state snapshots, return; there are no observations to make or update
        if not self.state_snapshots:
            logger.info("no state snapshots to create observation from")
            return None
        
        recent_snapshots = list(self.state_snapshots)[-context_size:]
        snapshot_texts = [s.to_inference_text() for s in recent_snapshots]
        last_observation = self.observations[-1] if self.observations else None

        # create new observation or update previous
        new_observation: Observation | None = None
        updated_observation: Observation | None = None # conditionally update
        if last_observation:
            # if we have a recent observation, either update it or create new
            # compare previous observation in the context of recent snapshots, try to update it
            updated_observation = await self.inferencer.update_observation(last_observation, snapshot_texts)
        if not updated_observation:
            # if we reach here, then either no last observation, or there is nothing meaningful to update
            # -> create new observation; look at recent snapshots and make a meaningful observation
            logger.info(f"No update to make, creating new observation from recent snapshots.")
            new_observation = await self.inferencer.create_new_observation(snapshot_texts)
            logger.info(f"Created new observation from recent snapshots: {new_observation.node if new_observation else None}")
            # NOTE: the parent caller will handle saving the cache-evicted observation to db
            return new_observation # may be None, which indicates no new, meaningful observation to make
        else:
            # there is a meaningful observation to update, so update local history and return None
            logger.info(f"Updated observation from recent snapshots: {updated_observation.node}")
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
        if self.observations:
            old_observation = self.observations[-1]
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
            
    def get_state_snapshots(
        self,
        limit: Optional[int] = None,
    ) -> List[UIStateSnapshot]:
        """
        Get state snapshots history.
        NOTE: only up to 50 recent snapshots are stored.

        Args:
            limit: Max snapshots to return

        Returns:
            List of UIStateSnapshot DTOs, most recent first.
        """
        snapshots = list(self.state_snapshots)

        if limit:
            snapshots = snapshots[-limit:]

        snapshots.reverse() # make the first snapshot the most recent

        return snapshots

    def get_observations(
        self,
        limit: Optional[int] = None,
    ) -> list[Observation]:
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

    # TODO: consider making these helpers be called during shutdown
    def clear_observations(self):
        """Clear observation history after persisting to DB."""
        self.observations.clear()

    def clear_state_snapshots(self):
        """Clear state snapshots history after persisting to DB."""
        self.state_snapshots.clear()

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
            "state_snapshots": len(self.state_snapshots),
            "observations": len(self.observations),
            "state_changes": len(self.recent_state_changes),
        }

    def start_background_tracking(self, poll_interval: float = 1.0):
        """
        Start tracking as a background task.
        Currently called by the background tracking endpoint.
            - Saves the latest polling interval.

        Args:
            poll_interval: How often to poll for changes (seconds)
        """
        if self._tracking_task is not None and self._tracking_task.done():
            self._tracking_task = None

        if self._tracking_task is not None and not self._tracking_task.done():
            raise RuntimeError("Observation tracking already running")

        self._tracking_task = asyncio.create_task(self.start_tracking(poll_interval))
        self.last_poll_interval = poll_interval # save only if new task is successfully started
        return self._tracking_task
    
    async def pause_tracking(self) -> bool:
        """
        Simply pause observation tracking, without resetting the internal states / history.
        NOTE: to re-activate tracking after pausing, run start_background_tracking() again.
        If background tracker is not running, does nothing.

        Returns: whether the tracker was previosuly running or not.
        """
        if self.running is True:
            self.running = False
            # pause briefly before returning to prevent race conditions w/ background tracking task
            await asyncio.sleep(0.1)
            return True
        # otherwise, just return, nothing to kill
        return False

    async def stop_tracking(self):
        """
        Stop observation tracking and wait for cleanup.
        Call this in lifespan shutdown.
        - Flushes the latest observation to DB and clears internal states / history.
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
                    pass # Expected

        # Clear the task reference after cleanup
        self._tracking_task = None

        # Flush the latest observation to db, since it's never saved by normal flow
        # NOTE: _save_observation() only persists the *previous* observation when a new one is created.
        if self.observations:
            last_observation = self.observations[-1]
            # NOTE: this uses a convenience wrapper that handles both embedding generation and saving
            # if we want to save to more than just the text log, should handle that here too.
            try:
                await self.embedding_generator.generate_and_save_embedding(
                    observation_id=last_observation.id,
                    observation_text=last_observation.node
                )
                logger.info(f"Flushed last observation to TEXT LOG on shutdown: {last_observation.node}")
            except Exception as e:
                logger.error(f"Failed to flush last observation on shutdown: {e}")

        # clear all internal states of previous tracking
        self.clear_observations()
        self.clear_state_snapshots()
        self.clear_state_changes()
    
    async def create_test_observation(self, context_size: int = 10) -> Optional[Observation]:
        """
        Creates a TEST observation object based on the current history of state snapshots.
        Verifies LLM / RAG functionality.
        """
        if not self.state_snapshots:
            logger.info("no state snapshots to create observation from")
            return None

        recent_snapshots = list(self.state_snapshots)[-context_size:]
        snapshot_texts = [s.to_inference_text() for s in recent_snapshots]

        # unconditional test — use helper to create observation
        new_observation = await self.inferencer.test_create_new_observation(state_snapshots=snapshot_texts)

        # return new observation (may be None)
        return new_observation

    async def replay_state_snapshots(self, state_snapshots: list[UIStateSnapshot]):
        """
        Replays a sequence of state snapshots through the observation pipeline.
        NOTE: allows mocked testing with predefined list of snapshot scenarios.
        """
        # pause tracking before replay to ensure no overrides and unexpected behavior
        previous_running = await self.pause_tracking()

        # loop over snapshots, and add each to the local snapshot history.
        for snapshot in state_snapshots:
            self.state_snapshots.append(snapshot)
            self.snapshot_counter += 1
            if self.snapshot_counter >= self.snapshot_context_size:
                new_observation = await self._create_or_update_observation(context_size=self.snapshot_context_size)
                if new_observation:
                    await self._save_observation(new_observation)
                self.snapshot_counter = 0

        # flush the last node
        # NOTE: add more saving logic here if we want more than just text log
        if self.observations:
            last_observation = self.observations[-1]
            await self.embedding_generator.generate_and_save_embedding(
                observation_id=last_observation.id,
                observation_text=last_observation.node
            )
            logger.info(f"Flushed last observation to TEXT LOG on replay end: {last_observation.node}")

        if previous_running:
            # resume tracking if previously running, using last poll interval
            self.start_background_tracking(poll_interval=self.last_poll_interval)
        