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

async def create_new_observation(
    actions: list[Action],
    latest_observation: Optional[Observation]
) -> Optional[Observation]:
    """
    Creates a NEW observation, or optionally None if no observation can be inferred.

    input: list of actions and latest observation
    output: observation node

    the logic for whether to use observation or not is handled elsewhere
    here, we just care about creating the right observation w/ semantic information

    this is the high-level helper, calling specialized agents to perform filtering and edge/node creation.
    """

    # for now, unconditional test
    # edge = await self.llm_client.acreate(
    #     # TODO: fill in prompt, schema
    # )
    new_observation = ShortTermPreferencesObservation(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        source_id="test_source_id",
        edge="test_edge",
        node="test_node",
        recurrence=1,
        importance=1.0
    )

