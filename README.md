# Portable-Brain
Your second brain living inside carry-on devices. Memory and context orchestration based on day-to-day [smartphone] HCI observations & habits (derived from low-level signals).

### How to Run (Locally):
- Use uvicorn + FastAPI set up to run the service locally.
- `poetry run uvicorn portable_brain.app:app --reload`

### DroidRun Client Connection
- Initialize set up with `droidrun setup` in terminal (one-time).
- Create an android virtual device (avd) - easiest to use Android Studio application.
- Start the android emulator with `emulator -avd <your_avd_name>`.
- Verify ADB connection with `adb devices` - emulator-5554 device should show up.
- Set up TCP forwarding (for local development w/ emulated android device) using `adb forward tcp:8001 tcp:8001`.

#### Install and set up Portal APK (DroidRun's android application that exposes A11Y tree information)
- adb -s emulator-5554 install path/to/portal.apk
- Navigate to `Settings > Accessibility > "Droidrun Portal"` on emulated Android device.
- Enable the application.
- Start the Portal application with `adb -s emulator-5554 shell am start-foreground-service com.android.portal/.PortalService`
- Verify that the Portal app is running with `adb -s emulator-5554 shell dumpsys accessibility | grep -i portal`.

### Dependencies
This project uses [poetry](https://python-poetry.org/) to manage dependencies. Please use `poetry add <dependency-group>` to add a new dependency to the project. If your dependencies are out-of-sync, use `poetry install` to fetch the latest version defined by `pyproject.toml` and `poetry.lock`.

### DroidRun Client + Tracker Implementation (scratch space currently)
#### DroidRun Client
general commands:
- execute_command (give natural language request)
- get_current_state (returns dict w/ state + raw a11y tree + elements)
- detect_state_change (check if device has changed since last check; outputs the diff)
- take_screenshot (screenshot bytes as PNG)
- get_installed_apps (list of installed apps)
- get_date (fetches current date and time on device)

fine controls:
- tap_by_index (element by idx)
- input_text (by index or the current element)
- swipe (from start coordinate to end coordinate, w/ duration)
- back (press back button)
- start_app (by package name)

action history:
- get_action_history (list of actions w/ timestamp, command, state changes)
    - for notable history, can customize ("change_type" in app_switch, screen_change, etc.)
    - **NOTE:** currently, only updated via execute_command. If user navigates UI on their own, it won't be reflected - see observation tracker below for user HCI tracking.
- clear_action_history

helpers:
- _serialize_state
- _classify_change (*determines the "change_type")

#### Observation Tracker 
(wraps droidrun client, allows background tasks to continuously observe user actions)
- start_tracking (method to continuously poll for UI state change and infer "events")
- _create_observation (helper to canonically format UI changes into events. To be updated w/ canonical DTO for event classification.)
- _infer_action (helper to classify event/action types from UI changes)
- get_observations (retrieves most recent observations from in-memory history)
- clear_observations
- start_background_tracking (wrapper to start tracking as a background task w/ async couroutine)
- stop_tracking (cleans up any currently running tasks and sets running status to false)

#### High-level Distinction + Tasks
1. Observation Tracker makes constant note of state change.
-> polls on the DroidRun client which uses raw client.get_state() to compare states, and return the change diff.
    - First pass raw state into making it a canonical state (1st layer of filtering) -> then, if canonical UI state diffs, we format the diff into canonical StateChange and return 
2. Then, if there is a state change, uses internal helper to create inferred action.
    - This internal helper should use canonical inferred action DTO

**LAST COMPLETION
3. The inferred action, alongside UI change diffs, is stored as a final observation (i.e. the nodes of memory representation).
    - TODO: the final observation should be universal for both observed/inferred actions and actions actually executed via commands from DroidRun Agent (TBD in future).
    - i.e. the observation should also become a DTO, just a wrapper to hold both the action (whether inferred or executed by user command) + ui state change (a.k.a. the outcome of action), and any supporting metadata.
**The above hierarchy of observations, state changes, actions, will help to keep a clean history of HCI data log.


**TODO: need a way to cleanly get metadata from UI states when referencing UI changes for specific apps. E.g. need to somehow store "username" for instagram UI changes so that it can be referenced in app.
- complete create_observation() helper
- complete filling in all inferred_action pathways
- also note the separation of concerns: droidrun client only tracks the latest two state changes (for diff) and observation tracker client keep a history of 10 most recent state changes for higher-level inference on possible actions / observations.


### Autonomous NL Query Eecution
- The service currently supports both Google GenAI client and Amazon Nova models.
- Current stage of autonomous execution is provided by the DroidRun client, which navigates and handles natural language query.
- TBD: memory service will eventually hold autonomous execution pipelines to handle processing on top of droidrun.

#### TODO:
- Experiment with Portal APK client for data parsing, implement filters
- Test natural language query capabilities for DroidRun Agent (w/ "Kevin" example)

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

<!-- ### Directory Organization:
- scripts/...
- tests/...
- src
    - portable_brain (package wrapper)
        - config
            - config.py
            - settings_mixins.py
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
- poetry dependencies (pyproject.toml, poetry.lock)
- .env -->
