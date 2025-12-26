# Portable-Brain
The second brain living inside your carry-on devices. Memory based on day-to-day smartphone HCI data.

TODO:
- **important**: changing integration from A11Y inspector service to DroidRun and AppAgent (both open-source)
- setting up communication with A11Y inspector service
- skeleton for agent and memory layer

Architecture (subject to change):
- FastAPI (Handles 2 main modes: background memory/KG updates + user request processing)
    - dependencies + lifetime management
        - session management
        - logging
    - LLM service
        - main orchestrator
        - retryable LLM client
            - primary
            - fallback
        - rate-limiter
        - prompts & pydantic validation
    - memory/db storage (each of the below data structures to be explored)
        - postgres
            - baseline (normalized)
                - events
                - interactions
            - base knowledge graph
                - edges
                - nodes
            - complex KGs
                - hypergraph
                - temporal graph
        - postgres + pgvector
            - vector embeddings
            - KG + vector embeddings
    - requests + communication w/ A11Y service
        - websocket connection
    - routes/API for commands by user

Directory Organization:
- scripts
- src
    - common
        - services
            - llm_service
            - kg/vector db services (future TBD)
            - a11y_listener (websocket client)
        - db
        - logging
            - logger.py
    - memory (background tasks)
        - baseline
        - knowledge_graph
        - vector_embeddings
        - hypergraph
        - temporal_graph
        - common
    - agent_service (user requests)
        - text
        - share_media
        - chat (answer questions, conversations, etc.)
        - orchestrator
        - common
            - types/...
    - core
        - lifespan.py
    - app.py
