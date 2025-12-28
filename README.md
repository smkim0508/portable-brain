# Portable-Brain
Your second brain living inside carry-on devices. Memory based on day-to-day [smartphone] HCI observations & habits.

#### TODO:
- set up poetry dependencies
- set up core lifespan and FastAPI app dependencies
- set up communication with DroidRun App (Python SDK)

### How to Run (Locally):
- Use uvicorn + FastAPI set up to run the service locally.
- `poetry run uvicorn portable_brain.app:app --reload`

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
- scripts/...
- tests/...
- src
    - config
        - .env
    - common
        - services
            - llm_service
                - rate_limiter
                - retryable_client
                - ...
            - kg_service/... (TBD)
            - vector_service/... (TBD)
            - droidrun_tools/... (Python SDK import)
        - db
            - models/...
            - crud/...
            - session.py
        - logging
            - logger.py
        - types/...
    - middleware
        - error_handler
    - memory (background tasks)
        - representations
            - baseline
            - knowledge_graph
            - vector_embeddings
            - hypergraph
            - temporal_graph
        - actions
            - update_memory
            - delete_memory
            - add_memory
        - common/...
    - agent_service (user requests)
        - api
            - routes/...
        - orchestrator
            - main_orchestrator
            - handlers
                - text
                    - llm_output_types/...
                    - system_prompts/...
                    - process_text
                - share_media
                    - llm_output_types/...
                    - system_prompts/...
                    - process_share_media
                - chat/... (answer questions, conversations, etc.)
                    - llm_output_types/...
                    - system_prompts/...
                    - process_conversation
        - common
            - types/... (specific to API handling + orchestrator LLM types)
    - monitoring
        - action_monitor/...
        - self_evaluator/...
        - semantic_filtering/...
    - core
        - lifespan.py
        - dependencies.py
    - app.py
    - poetry dependencies

### Dependencies
This project uses [poetry](https://python-poetry.org/) to manage dependencies. Please use `poetry add <dependency-group>` to add a new dependency to the project. If your dependencies are out-of-sync, use `poetry install` to fetch the latest version defined by `pyproject.toml` and `poetry.lock`.
