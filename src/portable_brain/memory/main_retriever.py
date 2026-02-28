# the main memory retriever layer/interface for the agent to interact with

from sqlalchemy.ext.asyncio import AsyncEngine
from typing import Optional
import numpy as np

from portable_brain.common.db.models.memory.structured_storage import StructuredMemory
from portable_brain.common.db.models.memory.text_embeddings import TextEmbeddingLogs
from portable_brain.common.db.models.memory.people import InterpersonalRelationship
from portable_brain.common.services.embedding_service.text_embedding.dispatcher import TypedTextEmbeddingClient

from portable_brain.common.logging.logger import logger

# data structures for caches
from collections import OrderedDict, deque

# structured memory fetch operations
from portable_brain.common.db.crud.memory.structured_memory_crud import (
    get_observations_by_memory_type,
    get_observations_by_entity,
    fulltext_search_observations,
    get_most_relevant_observations,
)
# text embeddings fetch operations
from portable_brain.common.db.crud.memory.text_embeddings_crud import (
    find_similar_embeddings,
    find_similar_texts,
    get_embedding_by_observation_id,
)
# people embeddings fetch operations
from portable_brain.common.db.crud.memory.people_crud import (
    get_person_by_id,
    find_person_by_name,
    find_similar_relationships,
)

class MemoryRetriever():
    """
    Implements helper methods for the retriever agent to access different memory.
    Handles internal operations like embedding natural language input to text embeddings, so the agent can freely call without overhead.
    - Tool-called by retriever agent.
    - Wraps CRUD read operations with semantically intuitive method names.
    - Each method specifies the memory type it targets.

    Caches:
    1. Exact match cache - skips embedding client entirely on identical query texts (LRU via OrderedDict, max 50)
    2. Semantic cache — skips db retrieval if a sufficiently similar query was seen before (FIFO deque, max 10)
    NOTE: only supported by find_semantically_similar for now, to be implemented for other methods
    - one caveat is if the memory is updated after the query, the cache may become stale; future TBD

    TODO: just baseline right now, memory to be refined.
    """

    def __init__(self, main_db_engine: AsyncEngine, text_embedding_client: TypedTextEmbeddingClient):
        self.main_db_engine = main_db_engine
        self.text_embedding_client = text_embedding_client
        # caches to reduce latency, for text embedding logs
        self._exact_cache: OrderedDict[str, list[str]] = OrderedDict()
        self._semantic_cache: deque[tuple[list[float], list[str]]] = deque(maxlen=10) # query_vector, results tuple
        self._cosine_similarity_threshold = 0.70 # threshold for semantic cache
        self._exact_cache_max = 50 # threshold for max number of items in exact query cache
        # exact name cache for find_person_by_name (keyed on normalized lowercase name)
        self._person_name_cache: OrderedDict[str, list[dict]] = OrderedDict()

    # =====================================================================
    # Structured Memory — Long-Term People (inter-personal relationships)
    # =====================================================================

    async def get_people_relationships(
        self,
        person_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[StructuredMemory]:
        """Retrieve long-term people observations (relationships, contacts)."""
        return await get_observations_by_memory_type(
            memory_type="long_term_people",
            main_db_engine=self.main_db_engine,
            target_entity_id=person_id,
            limit=limit,
        )

    # =====================================================================
    # Structured Memory — Long-Term Preferences (recurring user patterns)
    # =====================================================================

    async def get_long_term_preferences(
        self,
        source_app_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[StructuredMemory]:
        """Retrieve long-term preference observations (habitual usage patterns)."""
        return await get_observations_by_memory_type(
            memory_type="long_term_preferences",
            main_db_engine=self.main_db_engine,
            source_entity_id=source_app_id,
            limit=limit,
        )

    # =====================================================================
    # Structured Memory — Short-Term Preferences (recent user patterns)
    # =====================================================================

    async def get_short_term_preferences(
        self,
        source_app_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[StructuredMemory]:
        """Retrieve short-term preference observations (recent behavioural signals)."""
        return await get_observations_by_memory_type(
            memory_type="short_term_preferences",
            main_db_engine=self.main_db_engine,
            source_entity_id=source_app_id,
            limit=limit,
        )

    # =====================================================================
    # Structured Memory — Short-Term Content (recently viewed media/docs)
    # =====================================================================

    async def get_recent_content(
        self,
        source_id: Optional[str] = None,
        content_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[StructuredMemory]:
        """Retrieve short-term content observations (recently viewed documents/media)."""
        return await get_observations_by_memory_type(
            memory_type="short_term_content",
            main_db_engine=self.main_db_engine,
            source_entity_id=source_id,
            target_entity_id=content_id,
            limit=limit,
        )

    # =====================================================================
    # Structured Memory — Cross-type queries
    # =====================================================================

    async def get_all_observations_about_entity(
        self,
        entity_id: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[StructuredMemory]:
        """Find all observations mentioning a specific entity across all memory types."""
        return await get_observations_by_entity(
            entity_id=entity_id,
            main_db_engine=self.main_db_engine,
            entity_type=entity_type,
            limit=limit,
        )

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[tuple[StructuredMemory, float]]:
        """Full-text search across observation content. Returns (memory, rank) tuples."""
        return await fulltext_search_observations(
            search_query=query,
            main_db_engine=self.main_db_engine,
            memory_type=memory_type,
            limit=limit,
        )

    async def get_top_relevant_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[StructuredMemory]:
        """Retrieve highest-relevance observations ranked by importance * recurrence."""
        return await get_most_relevant_observations(
            main_db_engine=self.main_db_engine,
            memory_type=memory_type,
            limit=limit,
        )

    # =====================================================================
    # People Embeddings — Interpersonal relationship memory
    # =====================================================================

    async def get_person_by_id(
        self,
        person_id: str,
    ) -> Optional[InterpersonalRelationship]:
        """Look up a specific person's relationship record by their unique ID."""
        return await get_person_by_id(
            person_id=person_id,
            main_db_engine=self.main_db_engine,
        )

    async def find_person_by_name(
        self,
        name: str,
        similarity_threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """
        Fuzzy name lookup using trigram similarity. Handles typos, nicknames, and
        partial names. Returns dicts with full_name, relationship_description, and
        similarity_score, ordered by best match.

        NOTE: exact name cache keyed on normalized (lowercased) name, same LRU eviction as _exact_cache.
        """
        normalized = name.strip().lower()

        if normalized in self._person_name_cache:
            logger.info(f"Person name exact cache hit: {name}")
            self._person_name_cache.move_to_end(normalized)
            return self._person_name_cache[normalized]

        results = await find_person_by_name(
            name=name,
            main_db_engine=self.main_db_engine,
            similarity_threshold=similarity_threshold,
            limit=limit,
        )

        self._person_name_cache[normalized] = results
        if len(self._person_name_cache) > self._exact_cache_max:
            self._person_name_cache.popitem(last=False)
        return results

    async def find_similar_person_relationships(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[InterpersonalRelationship, float]]:
        """
        Semantic search over relationship descriptions using natural language.
        Embeds the query internally and returns (record, cosine_distance) tuples.
        """
        query_vectors = await self.text_embedding_client.aembed_text(text=[query])
        return await find_similar_relationships(
            query_vector=query_vectors[0],
            limit=limit,
            main_db_engine=self.main_db_engine,
        )

    # =====================================================================
    # Text Embeddings — Semantic similarity search
    # =====================================================================

    async def find_semantically_similar(
        self,
        query: str,
        limit: int = 5,
        distance_metric: str = "cosine",
        disable_cache: bool = False, # just for testing latency
    ) -> list[str]:
        """
        Semantic search across all embedded observations using natural language.
        Embeds the query internally. Returns a list of observation text strings ordered by similarity.

        NOTE: supports both exact match and semantic caches.
        """
        if disable_cache:
            # for testing, just skips cache logic entirely
            logger.info(f"Skipping cache for query: {query}")
            query_vectors = await self.text_embedding_client.aembed_text(text=[query], task_type="RETRIEVAL_QUERY")
            if not query_vectors:
                logger.warning(f"Failed to embed query: {query}, returning empty list")
                return []
            query_vector = query_vectors[0]

            results = await find_similar_texts(
                query_vector=query_vector,
                main_db_engine=self.main_db_engine,
                limit=limit,
                distance_metric=distance_metric
            )
            return results
        # 1) exact match — skip embedding entirely
        if query in self._exact_cache:
            logger.info(f"Exact cache hit: {query}")
            # if exact cache hit directly, just promote to most recently used spot in cache
            self._exact_cache.move_to_end(query)
            return self._exact_cache[query]
        
        query_vectors = await self.text_embedding_client.aembed_text(text=[query], task_type="RETRIEVAL_QUERY")
        if not query_vectors:
            logger.warning(f"Failed to embed query: {query}, returning empty list")
            return []
        query_vector = query_vectors[0]

        # 2) semantic cache — skip db retrieval if similar query was seen before
        # NOTE: current helper loops through all the cached vectors, but it is possible to implement this via numpy matrix multiplication to one-shot all cosine similarities
        # - above optimization not yet implemented since cache size is negligibly small (most case) and may be beneficial if recent cache computed first and returns
        semantic_cache_result = self._find_semantic_cache_hit(query_vector)
        if semantic_cache_result:
            logger.info(f"Semantic cache hit: {query}")
            # NOTE: if we have a semantic cache hit, we also promote to exact cache w/ similar vector results (not exact)
            # - this is logical as even without this promotion, the next query will be the same semantic cache hit anyways
            self._set_exact_cache(query, semantic_cache_result)
            return semantic_cache_result
        
        # 3) cache miss — retrieve from db and populate both caches
        results = await find_similar_texts(
            query_vector=query_vector,
            main_db_engine=self.main_db_engine,
            limit=limit,
            distance_metric=distance_metric
        )
        self._set_exact_cache(query, results)
        self._semantic_cache.append((query_vector, results))
        return results

    async def get_embedding_for_observation(
        self,
        observation_id: str,
    ) -> Optional[TextEmbeddingLogs]:
        """Look up the stored embedding for a specific observation by ID."""
        return await get_embedding_by_observation_id(
            observation_id=observation_id,
            main_db_engine=self.main_db_engine,
        )

    # =====================================================================
    # Utils for cache management
    # =====================================================================
    def _set_exact_cache(self, key: str, value: list[str]) -> None:
        """
        Simple helper to insert or or update elements in LRU exact cache, evicting the oldest entry if at capacity.
        """
        if key in self._exact_cache:
            self._exact_cache.move_to_end(key)
        self._exact_cache[key] = value
        if len(self._exact_cache) > self._exact_cache_max:
            self._exact_cache.popitem(last=False) # evict LRU

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Simple helper to compute cosine similarity between two vectors using numpy.
        Returns 0.0 if either vector has zero norm (undefined similarity).
        """
        va, vb = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def _find_semantic_cache_hit(self, query_vector: list[float]) -> Optional[list[str]]:
        """
        Simple helper to loop through semantic cache to find query hit via cos. sim. threshold.
        - Iterates newest-first so a more recent cached result is preferred over an older one.
        - If cache hit, returns just the cached similar text results
        - returns None if no semantic cache hit
        """
        for cached_vector, cached_results in reversed(self._semantic_cache):
            if self._cosine_similarity(query_vector, cached_vector) >= self._cosine_similarity_threshold:
                return cached_results
        return None
