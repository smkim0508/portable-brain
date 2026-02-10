# CRUD operations for text embeddings with pgvector
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import select, text
from portable_brain.common.db.models.memory.text_embeddings import TextEmbeddingLogs
from portable_brain.common.db.session import get_async_session_maker
from portable_brain.common.logging.logger import logger
from datetime import datetime
from typing import Optional

# TODO: to be updated, for now a simple test crud
async def save_text_embedding(
    observation_id: str,
    observation_text: str,
    embedding_vector: list[float],
    main_db_engine: AsyncEngine,
    created_at: Optional[datetime] = None
) -> None:
    """
    Save a text embedding to the database.

    Args:
        observation_id: Unique identifier for the observation
        observation_text: The original text that was embedded
        embedding_vector: The embedding vector (list of floats)
        main_db_engine: Async database engine
        created_at: Optional timestamp (defaults to now)
    """
    session_maker = get_async_session_maker(main_db_engine)

    try:
        async with session_maker() as session:
            embedding = TextEmbeddingLogs(
                id=observation_id,
                observation_text=observation_text,
                embedding_vector=embedding_vector,
                observation_id=observation_id,
                created_at=created_at or datetime.now()
            )
            session.add(embedding)
            await session.commit()
            logger.info(f"Saved text embedding for observation {observation_id}")
    except Exception as e:
        logger.error(f"Failed to save text embedding: {e}")
        raise


async def find_similar_embeddings(
    query_vector: list[float],
    limit: int,
    main_db_engine: AsyncEngine,
    distance_metric: str = "cosine"  # "cosine", "l2", or "inner_product"
) -> list[tuple[TextEmbeddingLogs, float]]:
    """
    Find the most similar embeddings using vector similarity search.

    Args:
        query_vector: The query embedding vector
        limit: Maximum number of results to return
        main_db_engine: Async database engine
        distance_metric: Distance metric to use ("cosine", "l2", or "inner_product")

    Returns:
        List of tuples (TextEmbedding, distance)
    """
    session_maker = get_async_session_maker(main_db_engine)

    # Choose distance function based on metric
    distance_functions = {
        "cosine": TextEmbeddingLogs.embedding_vector.cosine_distance,
        "l2": TextEmbeddingLogs.embedding_vector.l2_distance,
        "inner_product": TextEmbeddingLogs.embedding_vector.max_inner_product
    }

    if distance_metric not in distance_functions:
        raise ValueError(f"Invalid distance metric: {distance_metric}. Use 'cosine', 'l2', or 'inner_product'")

    distance_func = distance_functions[distance_metric]

    try:
        async with session_maker() as session:
            # Build query with distance calculation
            stmt = (
                select(
                    TextEmbeddingLogs,
                    distance_func(query_vector).label("distance")
                )
                .order_by("distance")
                .limit(limit)
            )

            result = await session.execute(stmt)
            results = result.all()

            logger.info(f"Found {len(results)} similar embeddings")
            return [(row[0], row[1]) for row in results]

    except Exception as e:
        logger.error(f"Failed to find similar embeddings: {e}")
        raise


async def get_embedding_by_id(
    observation_id: str,
    main_db_engine: AsyncEngine
) -> Optional[TextEmbeddingLogs]:
    """
    Retrieve a text embedding by observation ID.

    Args:
        observation_id: The observation identifier
        main_db_engine: Async database engine

    Returns:
        TextEmbedding or None if not found
    """
    session_maker = get_async_session_maker(main_db_engine)

    try:
        async with session_maker() as session:
            stmt = select(TextEmbeddingLogs).where(TextEmbeddingLogs.id == observation_id)
            result = await session.execute(stmt)
            embedding = result.scalar_one_or_none()
            return embedding
    except Exception as e:
        logger.error(f"Failed to get embedding by ID: {e}")
        raise


async def delete_embedding(
    observation_id: str,
    main_db_engine: AsyncEngine
) -> bool:
    """
    Delete a text embedding by observation ID.

    Args:
        observation_id: The observation identifier
        main_db_engine: Async database engine

    Returns:
        True if deleted, False if not found
    """
    session_maker = get_async_session_maker(main_db_engine)

    try:
        async with session_maker() as session:
            stmt = select(TextEmbeddingLogs).where(TextEmbeddingLogs.id == observation_id)
            result = await session.execute(stmt)
            embedding = result.scalar_one_or_none()

            if embedding:
                await session.delete(embedding)
                await session.commit()
                logger.info(f"Deleted text embedding for observation {observation_id}")
                return True
            else:
                logger.warning(f"No embedding found for observation {observation_id}")
                return False
    except Exception as e:
        logger.error(f"Failed to delete embedding: {e}")
        raise
