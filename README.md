# Portable-Brain
Your second brain living inside carry-on devices. Memory based on day-to-day [smartphone] HCI observations & habits.

#### TODO:
- **important**: changing integration from A11Y inspector service to DroidRun and AppAgent (both open-source)
- setting up communication with DroidRun App (Python SDK)
- skeleton for agent and memory layer

### Architecture (subject to change):
- FastAPI (Handles 2 main modes: background memory/KG updates + user request processing)
    - dependencies + lifetime management
        - session management
        - logging
    - LLM service
        - main orchestrator
            - self-evaluator (based on user response to suggested action)
                - evaluated results sent to special memory handler to update directly, or observe future user action
            - memory updates
        - monitoring agent
            - semantic filtering on top of high-potential signals given by DroidRun app
            - memory updates
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
    - requests + communication w/ DroidRun Service
        - simply through Python SDK import
        - passes through monitoring agent, any notable actions are forwarded to memory handler
    - routes/API for commands by user

### Directory Organization:
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
