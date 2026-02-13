# crud for structured memory in db

# sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

# Canonical DTOs for db model and observation
from portable_brain.common.db.models.memory.structured_storage import StructuredMemory
from portable_brain.monitoring.background_tasks.types.observation.observations import (
    Observation,
    LongTermPeopleObservation,
    LongTermPreferencesObservation,
    ShortTermPreferencesObservation,
    ShortTermContentObservation,
)
# async sessionmaker
from portable_brain.common.db.session import get_async_session_maker
# logger
from portable_brain.common.logging.logger import logger

async def save_observation_to_structured_memory(observation: Observation, main_db_engine: AsyncEngine) -> None:
    """
    Helper to save observation node to structured memory in SQL db.
    - Parses Observation DTO into StructuredMemory ORM based on observation subtype.
    - Uses async sessionmaker to create session.
    - SQLAlchemy allows ORM mapped operations.
    """

    # Parse Observation DTO into StructuredMemory ORM by subtype case work
    if isinstance(observation, LongTermPeopleObservation):
        orm_obj = StructuredMemory(
            id=observation.id,
            memory_type=observation.memory_type.value,
            node_content=observation.node,
            edge_type=observation.edge,
            source_entity_id="me",
            source_entity_type="user",
            target_entity_id=observation.target_id,
            target_entity_type="person",
            created_at=observation.created_at,
            updated_at=observation.created_at,
            importance=observation.importance,
            recurrence=1,
        )
    elif isinstance(observation, (LongTermPreferencesObservation, ShortTermPreferencesObservation)):
        orm_obj = StructuredMemory(
            id=observation.id,
            memory_type=observation.memory_type.value,
            node_content=observation.node,
            edge_type=observation.edge,
            source_entity_id=observation.source_id,
            source_entity_type="app",
            target_entity_id=None,
            target_entity_type=None,
            created_at=observation.created_at,
            updated_at=observation.created_at,
            importance=observation.importance,
            recurrence=observation.recurrence,
        )
    elif isinstance(observation, ShortTermContentObservation):
        orm_obj = StructuredMemory(
            id=observation.id,
            memory_type=observation.memory_type.value,
            node_content=observation.node,
            edge_type=None,
            source_entity_id=observation.source_id,
            source_entity_type="content_source",
            target_entity_id=observation.content_id,
            target_entity_type="content",
            created_at=observation.created_at,
            updated_at=observation.created_at,
            importance=observation.importance,
            recurrence=1,
        )
    else:
        logger.error(f"Unsupported observation type: {type(observation)}")
        raise TypeError(f"Unsupported observation type: {type(observation)}")

    session_maker = get_async_session_maker(main_db_engine)
    try:
        async with session_maker() as session:
            session.add(orm_obj)
            await session.commit()
            logger.info(f"Saved observation {observation.id} to structured memory")
    except Exception as e:
        logger.error(f"Failed to save observation to structured memory: {e}")
        raise
