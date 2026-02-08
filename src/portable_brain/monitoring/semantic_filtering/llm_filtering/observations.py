# helpers to create, classify, update, and filter observations
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum
import uuid
from datetime import datetime

# action and observation DTOs
from portable_brain.monitoring.background_tasks.types.observation.observations import (
    Observation,
    LongTermPeopleObservation,
    LongTermPreferencesObservation,
    ShortTermPreferencesObservation,
    ShortTermContentObservation
)
from portable_brain.monitoring.background_tasks.types.action.actions import Action

from portable_brain.monitoring.observation_repository import ObservationRepository

# for inference
from portable_brain.monitoring.semantic_filtering.llm_filtering.system_prompts.observation_prompts import ObservationPrompts
from portable_brain.monitoring.semantic_filtering.llm_filtering.llm_response_types.observation_responses import TestObservationLLMResponse, NewObservationLLMResponse, UpdatedObservationLLMResponse

# logger
from portable_brain.common.logging.logger import logger

class ObservationInferencer(ObservationRepository):
    """
    Helper to inference observations from history of actions and recent state changes.
    NOTE: inherits from repository for dependencies.
    """

    async def test_create_new_observation(self, actions: list[Action]) -> Optional[Observation]:
        test_llm_response = await self.llm_client.acreate(
            system_prompt=ObservationPrompts.test_system_prompt,
            user_prompt=ObservationPrompts.test_user_prompt,
            response_model=TestObservationLLMResponse
        )
        # TODO: parse response
        logger.info(f"llm response: {test_llm_response}")
        return # returns nothing, just check llm response via log

    async def create_new_observation(
        self,
        actions: list[Action]
    ) -> Optional[Observation]:
        """
        Creates a NEW observation, or optionally None if no observation can be inferred.

        input: list of actions and latest observation
        output: observation node

        the logic for whether to use observation or not is handled elsewhere
        here, we just care about creating the right observation w/ semantic information

        this is the high-level helper, calling specialized agents to perform filtering and edge/node creation.
        """

        new_observation_response: NewObservationLLMResponse = await self.llm_client.acreate(
            system_prompt=ObservationPrompts.create_new_observation_system_prompt,
            user_prompt=ObservationPrompts.get_create_new_observation_user_prompt(actions),
            response_model=NewObservationLLMResponse
        )

        # parse response and log
        observation_node = new_observation_response.observation_node
        observation_reasoning = new_observation_response.reasoning
        logger.info(f"new observation llm response: {observation_node}, reasoning: {observation_reasoning}")
        
        if not observation_node:
            return None
        
        # if there is an observation to be made, classify it.
        
        # format into observation
        # TODO: the observation type should depend on the observation node, possibly inferenced together.
        new_observation = ShortTermPreferencesObservation(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            source_id="test_source_id", # to be updated
            edge=None,
            node=observation_node,
            recurrence=1,
            importance=1.0
        )
        # TODO: need some way to fetch observation and update edge, node details, and recurrence info

        return new_observation
    
    async def update_observation(self, observation: Observation, actions: list[Action]) -> Optional[Observation]:
        """
        Updates an old observation OR returns None if no meaningful observation can be inferred.
        """
        # TODO: complete this

        updated_observation_response = await self.llm_client.acreate(
            system_prompt=ObservationPrompts.update_existing_observation_system_prompt,
            user_prompt=ObservationPrompts.get_update_observation_user_prompt(observation, actions),
            response_model=UpdatedObservationLLMResponse
        )

        # parse response and log
        updated_observation_node = updated_observation_response.updated_observation_node
        reasoning = updated_observation_response.reasoning
        logger.info(f"new observation llm response: {updated_observation_node}, reasoning: {reasoning}")

        # if no meaningful observation can be inferred, return None
        if not updated_observation_node:
            return None

        # otherwise, form observation to return
        # TODO: classification of observation type is needed
        updated_observation = ShortTermPreferencesObservation(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            source_id="test_source_id", # to be updated
            edge=None,
            node=updated_observation_node,
            recurrence=1,
            importance=1.0
        )
        return updated_observation
