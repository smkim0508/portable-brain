# tool calling declarations for memory retriever functions

# NOTE: each declaration maps 1:1 to a MemoryRetriever method
# TODO: should be refactored once memory is updated

# =====================================================================
# Structured Memory — Long-Term People
# =====================================================================

get_people_relationships_declaration = {
    "name": "get_people_relationships",
    "description": "Retrieve long-term people observations about inter-personal relationships and contacts. Use this to look up what the user knows about a specific person or to list all known relationship observations.",
    "parameters": {
        "type": "object",
        "properties": {
            "person_id": {
                "type": "string",
                "description": "Optional unique identifier of the target person to filter by, e.g. 'sarah_smith'. Omit to retrieve all people observations.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": [],
    },
}

# =====================================================================
# Structured Memory — Long-Term Preferences
# =====================================================================

get_long_term_preferences_declaration = {
    "name": "get_long_term_preferences",
    "description": "Retrieve long-term preference observations representing habitual usage patterns. Use this to recall recurring user habits like app usage sequences or established routines.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_app_id": {
                "type": "string",
                "description": "Optional app package identifier to filter preferences by, e.g. 'com.instagram.android'. Omit to retrieve all long-term preferences.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": [],
    },
}

# =====================================================================
# Structured Memory — Short-Term Preferences
# =====================================================================

get_short_term_preferences_declaration = {
    "name": "get_short_term_preferences",
    "description": "Retrieve short-term preference observations representing recent behavioural signals. Use this to recall the user's recent preferences that may not yet be established as long-term habits.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_app_id": {
                "type": "string",
                "description": "Optional app package identifier to filter preferences by, e.g. 'com.slack.android'. Omit to retrieve all short-term preferences.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": [],
    },
}

# =====================================================================
# Structured Memory — Short-Term Content
# =====================================================================

get_recent_content_declaration = {
    "name": "get_recent_content",
    "description": "Retrieve short-term content observations about recently viewed documents, media, or other content. Use this to recall what the user was recently looking at.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "Optional identifier of the content source to filter by, e.g. an app or platform ID.",
            },
            "content_id": {
                "type": "string",
                "description": "Optional identifier of the specific content item to look up.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": [],
    },
}

# =====================================================================
# Structured Memory — Cross-type queries
# =====================================================================

get_all_observations_about_entity_declaration = {
    "name": "get_all_observations_about_entity",
    "description": "Find all observations mentioning a specific entity across all memory types. Use this when you need a complete picture of everything known about a particular person, app, or object.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The entity identifier to search for, e.g. 'sarah_smith' or 'com.instagram.android'.",
            },
            "entity_type": {
                "type": "string",
                "description": "Optional filter by entity type: 'person', 'app', 'content_source', 'workspace', 'channel', etc.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": ["entity_id"],
    },
}

search_memories_declaration = {
    "name": "search_memories",
    "description": "Full-text search across all observation content using natural language. Use this when you need to find memories matching a keyword or phrase, like 'dinner plans' or 'project deadline'.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query to match against observation content.",
            },
            "memory_type": {
                "type": "string",
                "description": "Optional filter by memory type: 'long_term_people', 'long_term_preferences', 'short_term_preferences', or 'short_term_content'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": ["query"],
    },
}

get_top_relevant_memories_declaration = {
    "name": "get_top_relevant_memories",
    "description": "Retrieve the highest-relevance observations ranked by importance and recurrence. Use this to get the most significant memories the user has, optionally filtered by memory type.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_type": {
                "type": "string",
                "description": "Optional filter by memory type: 'long_term_people', 'long_term_preferences', 'short_term_preferences', or 'short_term_content'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
            },
        },
        "required": [],
    },
}

# =====================================================================
# Text Embeddings — Semantic similarity search
# =====================================================================

find_semantically_similar_declaration = {
    "name": "find_semantically_similar",
    "description": "Semantic similarity search across all embedded observations using natural language. Use this when full-text search is insufficient and you need meaning-based retrieval. Embedding is handled internally.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search semantically against stored observation embeddings.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
            },
            "distance_metric": {
                "type": "string",
                "description": "Distance metric for similarity: 'cosine', 'l2', or 'inner_product'. Defaults to 'cosine'.",
            },
        },
        "required": ["query"],
    },
}

get_embedding_for_observation_declaration = {
    "name": "get_embedding_for_observation",
    "description": "Look up the stored text embedding for a specific observation by its ID. Use this to retrieve the raw embedding vector and original text of a known observation.",
    "parameters": {
        "type": "object",
        "properties": {
            "observation_id": {
                "type": "string",
                "description": "The unique identifier of the observation whose embedding to retrieve.",
            },
        },
        "required": ["observation_id"],
    },
}

# aggregated list of all memory retriever declarations to be used by the agent
memory_retriever_declarations = [
    get_people_relationships_declaration,
    get_long_term_preferences_declaration,
    get_short_term_preferences_declaration,
    get_recent_content_declaration,
    get_all_observations_about_entity_declaration,
    search_memories_declaration,
    get_top_relevant_memories_declaration,
    find_semantically_similar_declaration,
    get_embedding_for_observation_declaration,
]
