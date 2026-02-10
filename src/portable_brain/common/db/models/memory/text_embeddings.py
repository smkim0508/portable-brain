# ORM model for storing text embeddings with pgvector
from portable_brain.common.db.models.base import MainDB_Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Index
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional

# TODO: to be updated, for now a simple test
class TextEmbeddingLogs(MainDB_Base):
    """
    Store text embeddings for semantic search of observation nodes.
    Uses pgvector for efficient similarity search.

    Retrieval possibilities:
    - Similarity search: Query by embedding vector using cosine/L2 distance
    - Text lookup: Query by observation_id or observation_text
    - Temporal: Query by created_at

    Usage:
        # Insert embedding
        embedding = TextEmbedding(
            id="obs_123",
            observation_text="User frequently messages sarah_smith on Instagram",
            embedding_vector=[0.1, 0.2, ...],  # 768-dim for gemini-embedding-001
            created_at=datetime.now()
        )

        # Similarity search
        results = session.query(TextEmbedding).order_by(
            TextEmbedding.embedding_vector.l2_distance(query_vector)
        ).limit(5).all()
    """
    __tablename__ = "text_embeddings"

    # Primary key
    id: Mapped[str] = mapped_column(String, primary_key=True)
    # Unique identifier for the observation (could be observation.id)

    # Original text that was embedded
    observation_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The actual observation node text

    # Embedding vector (768 dimensions for gemini-embedding-001)
    # Adjust dimension based on your embedding model
    embedding_vector: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False, default=datetime.now)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=datetime.now, nullable=True)

    # Optional: reference to original observation if needed
    observation_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)

    # Indexes for efficient queries
    __table_args__ = (
        # For timestamp-based queries
        Index('idx_text_embeddings_created', 'created_at'),

        # For observation lookup
        Index('idx_text_embeddings_obs_id', 'observation_id'),

        # HNSW index for fast vector similarity search
        # Using cosine distance (can also use 'vector_l2_ops' for L2 distance)
        Index(
            'idx_text_embeddings_vector_cosine',
            'embedding_vector',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding_vector': 'vector_cosine_ops'}
        ),
    )
