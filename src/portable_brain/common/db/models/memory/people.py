# representation of interpersonal relationships as flat embeddings
from portable_brain.common.db.models.base import MainDB_Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, Index, Computed
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional


class InterpersonalRelationship(MainDB_Base):
    """
    NOTE: just the baseline interpersonal relationship model, to be updated and cleaned up in the future

    Store interpersonal relationships between the user and people they interact with
    via messages, social media, email, or any other channel.

    Name Lookup Design — Trigram similarity over lexical embedding:
        Names are proper nouns; they carry no semantic meaning that dense or lexical
        embeddings can exploit beyond character patterns. A lexical (sparse/n-gram)
        embedding is functionally equivalent to trigram similarity, but PostgreSQL's
        pg_trgm extension provides this natively via a GIN index with no external
        model call needed at query time. The similarity() operator returns a 0-1
        score suitable for direct threshold filtering:

            WHERE similarity(full_name, 'John Smith') > 0.5

        This handles typos ("Jon Smith"), nicknames ("Johnny"), and mononyms
        more practically than a vector distance over a sparse embedding.

        Requires: CREATE EXTENSION IF NOT EXISTS pg_trgm;

    The relationship_description uses a dense vector for semantic similarity search
    over natural language content, which is where embeddings provide real value.

    Retrieval possibilities:
        - Fuzzy name lookup:        WHERE full_name % 'John Smith' (trigram %)
                                    WHERE similarity(full_name, 'John Smith') > 0.5
        - Semantic relationship:    ORDER BY relationship_vector <=> query_vec (cosine)
        - Full-text description:    WHERE search_vector @@ to_tsquery('close friend')
        - Platform-scoped lookup:   WHERE platform = 'instagram' AND platform_handle = '@john'
        - Recency / frequency:      ORDER BY last_interacted_at DESC, interaction_count DESC

    Usage:
        # Insert
        rel = InterpersonalRelationship(
            id="person_uuid_123",
            first_name="Sarah",
            last_name="Smith",
            full_name="Sarah Smith",
            relationship_description="Close friend from work; shares interest in music...",
            relationship_vector=[0.1, 0.2, ...],  # embed relationship_description
        )

        # Fuzzy name lookup
        from sqlalchemy import func
        results = session.query(InterpersonalRelationship).filter(
            func.similarity(InterpersonalRelationship.full_name, "Sara Smith") > 0.4
        ).all()

        # Semantic search over relationship descriptions
        results = session.query(InterpersonalRelationship).order_by(
            InterpersonalRelationship.relationship_vector.cosine_distance(query_vector)
        ).limit(5).all()
    """
    __tablename__ = "interpersonal_relationships"

    # Primary key
    id: Mapped[str] = mapped_column(String, primary_key=True)

    # --- Person identity ---
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    last_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Denormalized full name used for trigram GIN index — set explicitly at write time.
    # For mononyms, mirrors first_name.
    full_name: Mapped[str] = mapped_column(String, nullable=False)

    # Optional platform context for same-name disambiguation
    platform: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # e.g. "instagram", "email", "sms", "slack"
    platform_handle: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # e.g. "@sarah_smith", "sarah@example.com"

    # --- Relationship content ---
    # Semantic, natural language summary of the relationship (LLM-generated)
    relationship_description: Mapped[str] = mapped_column(Text, nullable=False)

    # Dense embedding of relationship_description for semantic similarity search.
    # 1536 dimensions — matches gemini-embedding-001 / text-embedding-3-small.
    relationship_vector: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)

    # Full-text search on relationship_description
    search_vector: Mapped[Optional[str]] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', relationship_description)", persisted=True)
    )

    # --- Temporal metadata ---
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )
    last_interacted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Interaction frequency — incremented each time the system records a new interaction
    interaction_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    __table_args__ = (
        # Trigram GIN index for fuzzy name lookup via similarity() or the % operator.
        # Requires pg_trgm extension.
        Index(
            'idx_interpersonal_name_trgm',
            'full_name',
            postgresql_using='gin',
            postgresql_ops={'full_name': 'gin_trgm_ops'}
        ),

        # HNSW index for fast cosine similarity search on relationship descriptions.
        Index(
            'idx_interpersonal_vector_cosine',
            'relationship_vector',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'relationship_vector': 'vector_cosine_ops'}
        ),

        # GIN index for full-text search on relationship descriptions.
        Index('idx_interpersonal_search_vector', 'search_vector', postgresql_using='gin'),

        # Platform-scoped person lookup (disambiguate same-name people across platforms).
        Index('idx_interpersonal_platform', 'platform', 'platform_handle'),

        # Recency queries.
        Index('idx_interpersonal_recency', 'last_interacted_at'),
    )
