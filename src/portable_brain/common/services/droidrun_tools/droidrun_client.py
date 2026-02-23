"""
DroidRun SDK Client Wrapper for Memory Agent Integration

This wrapper provides:
1. High-level command execution via DroidAgent
2. Low-level device monitoring via AdbTools
3. Action tracking for memory updates
4. State change detection

Based on DroidRun SDK exploration, available components:
- AdbTools: Direct device control (tap, swipe, input_text, get_state, etc.)
- DroidAgent: LLM-based orchestration (natural language commands)
- PortalClient: Low-level Portal communication
- Events: TapActionEvent, SwipeActionEvent, InputTextActionEvent, etc.
"""

import os
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from functools import wraps
import re
import uuid

# DroidRun SDK imports
from droidrun import DroidAgent, AdbTools, DroidrunConfig, DeviceConfig, AgentConfig
from droidrun.agent import ResultEvent
from droidrun import load_llm

# settings
from portable_brain.config.app_config import get_service_settings
# logging
from portable_brain.common.logging.logger import logger

# Canonical DTOs for UI state, inferred action, observations
from portable_brain.monitoring.background_tasks.types.ui_states.ui_state import UIState, UIActivity
from portable_brain.monitoring.background_tasks.types.ui_states.state_changes import UIStateChange, StateChangeSource
from portable_brain.monitoring.background_tasks.types.ui_states.state_change_types import StateChangeType
from portable_brain.monitoring.background_tasks.types.action.action_types import ActionType
from portable_brain.monitoring.background_tasks.types.action.actions import (
    Action,
    AppSwitchAction,
    InstagramMessageSentAction,
    InstagramPostLikedAction,
    WhatsAppMessageSentAction,
    SlackMessageSentAction,
    # TBD
)
# Execution Result DTO
from portable_brain.common.services.droidrun_tools.common.execution_types import ExecutionResult, RawExecutionResult

# Tree parser
from portable_brain.common.services.droidrun_tools.a11y_tree_parser import denoise_formatted_text

def ensure_connected(func):
    """
    Decorator to ensure DroidRun client is connected before executing method.
    Automatically reconnects if connection was lost, and raises an error if reconnection fails.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self._connected:
            logger.warning(f"DroidRun client disconnected, attempting to reconnect...")
            await self.connect()
            if not self._connected:
                logger.error(f"Failed to reconnect to device {self.device_serial}")
                raise ConnectionError(f"Failed to reconnect to device {self.device_serial}")
        return await func(self, *args, **kwargs)
    return wrapper

class DroidRunClient:
    """
    Wrapper for DroidRun SDK to integrate with FastAPI memory agent service.

    Provides two interfaces:
    1. High-level: Execute enriched natural language commands via DroidAgent (requires LLM init)
    2. Low-level: Monitor device state and actions for memory updates (does NOT need LLM)
    """

    def __init__(
        self,
        device_serial: str = "emulator-5554",
        use_tcp: bool = True,
        llm_instance = None, # optionally, pass in an LLM instance instead of initializing here
        api_key: Optional[str] = None # can use observation-only without LLM
    ):
        """
        Initialize DroidRun client.

        Args:
            device_serial: Android device serial number
            use_tcp: Use TCP mode for faster Portal communication (default: True)
            llm_instance: LLM instance for DroidAgent (if None, will use load_llm)
            api_key: Google API key; optional, since observation does not require LLM initialization.
            NOTE: still highly recommended to initialize service with DroidRun's internal LLM to execute commands.
        """
        self.device_serial: str = device_serial
        self.use_tcp: bool = use_tcp
        self.disable_llm: bool = False

        # Initialize LLM: use provided instance or load from DroidRun
        if llm_instance is not None:
            self.llm = llm_instance
        else:
            # Use DroidRun's load_llm with Google provider
            # Set API key in environment if provided
            self.llm = None # Initialize to None first

            if api_key:
                # NOTE: DroidRun Client expects precisely "GOOGLE_API_KEY" in the environment, so we set it explicitly.
                os.environ['GOOGLE_API_KEY'] = api_key

            # Only attempt to load LLM if API key is available in environment
            # This prevents partial initialization of Google GenAI client
            if os.environ.get('GOOGLE_API_KEY'):
                try:
                    self.llm = load_llm(
                        provider_name="GoogleGenAI",
                        model="gemini-2.5-flash-lite"
                    )
                    logger.info(f"Successfully initialized LLM agent for DroidRun!")
                except Exception as e:
                    logger.error(f"LLM client initialization failed for DroidAgent, error: {e}.\nYou will not be able to execute commands, OBSERVATION ONLY!")
                    self.llm = None # Ensure it's None, not a partial client
                    self.disable_llm = True
            else:
                logger.warning(f"No LLM API key found in environment. DroidAgent disabled, OBSERVATION ONLY!")
                self.disable_llm = True

        # Initialize AdbTools for monitoring and direct control
        self.tools = AdbTools(
            serial=device_serial,
            use_tcp=use_tcp,
        )

        # Configure device
        self.device_config = DeviceConfig(
            serial=device_serial,
            use_tcp=use_tcp,
        )

        # Track last known state for change detection
        self.last_state: UIState | None = None
        self.execution_history: list[ExecutionResult] = []

        self._connected = False

    async def connect(self) -> bool:
        """
        Establish connection to device and Portal.

        Returns:
            True if connection successful
        """
        try:
            await self.tools.connect()

            # Test connection with ping
            raw_state = await self.tools.get_state()
            state = self._format_raw_ui_state(raw_state)
            self.last_state = state
            self._connected = True

            logger.info(f"Connected to device {self.device_serial}")
            logger.info(f"Current app package: {state.package}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to device: {e}")
            self._connected = False
            return False

    # =====================================================================
    # HIGH-LEVEL INTERFACE: Execute Enriched Commands
    # =====================================================================

    @ensure_connected
    async def execute_command(
        self,
        enriched_command: str,
        reasoning: bool = False,
        timeout: int = 120, # unused right now
    ) -> RawExecutionResult:
        """
        Execute enriched natural language command via DroidAgent.

        This is the main interface for your memory agent to send commands
        after semantic enrichment and entity resolution.

        Args:
            enriched_command: Natural language command (already enriched)
                Example: "Open Messages app, send SMS to Kevin Chen (+1-234-567-8900)
                         with message 'about dinner'"
            reasoning: Enable planning mode (default: False for simple tasks)
            timeout: Command timeout in seconds

        Returns:
            RawExecutionResult with:
            - timestamp: datetime
            - command: str
            - success: bool
            - reason: str (explanation or answer)
            - steps: int (number of steps taken)
            - structured_output: Optional[BaseModel] (if output_model was provided)

        NOTE: not to be confused with ExecutionResult, which has additional metadata.
        - For internal usage, but not exposed yet to ExecutionAgent
        """
        if not self.llm or self.disable_llm:
            raise ValueError("LLM instance required for execute_command() - OBSERVATION ONLY MODE.")

        # Capture state before execution
        state_before_raw = await self.tools.get_state()
        state_before = self._format_raw_ui_state(state_before_raw)

        # Configure agent
        agent_config = AgentConfig(reasoning=reasoning)
        config = DroidrunConfig(
            device=self.device_config,
            agent=agent_config,
        )

        # Create and run agent
        agent = DroidAgent(
            goal=enriched_command,
            llms=self.llm,
            tools=self.tools,
            config=config,
        )

        result = await agent.run()

        # Capture state after execution
        state_after_raw = await self.tools.get_state()
        state_after = self._format_raw_ui_state(state_after_raw)

        # Record full execution result to history
        full_execution_result = ExecutionResult(
            timestamp=datetime.now(),
            command=enriched_command,
            success=result.success,
            reason=result.reason,
            steps=result.steps,
            state_before=state_before,
            state_after=state_after,
            change_type=self._classify_change(state_before, state_after),
            structured_output=result.structured_output,
        )
        self.execution_history.append(full_execution_result)

        # update states
        self.last_state = state_after

        # NOTE: returns a more minimal result to agent for now.
        # TODO: once execution agent becomes multi-turn, consider appending full context.
        return RawExecutionResult(
            timestamp=datetime.now(),
            command=enriched_command,
            success=result.success,
            reason=result.reason,
            steps=result.steps,
            structured_output=result.structured_output
        )

    # =====================================================================
    # LOW-LEVEL INTERFACE: Monitor Device State
    # =====================================================================

    @ensure_connected
    async def get_date(self) -> str:
        """
        Retrieves the current time and date on device.
        """
        return await self.tools.get_date()
    
    @ensure_connected
    async def get_raw_state(self):
        """
        Retrieves raw state from device.
        """
        return await self.tools.get_state()
    
    @ensure_connected
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get current device state with UI tree and phone context.

        Returns:
            Dictionary with:
                - formatted_text: str (indexed UI description)
                - focused_element: str (currently focused element)
                - ui_elements: List[Dict] (flattened UI tree with indices)
                - phone_state: Dict (package, activity, editable status)
                - raw_tree: Dict (unfiltered accessibility tree)

        Example:
            state = await client.get_current_state()
            print(f"Current app: {state['phone_state']['packageName']}")
            print(f"UI elements: {len(state['ui_elements'])}")
        """

        # Get processed state
        # NOTE: the DroidRun SDK returns as a tuple, so we unpack and handle it as a dict for more reliable downstream processing
        formatted_text, focused_element, ui_elements, phone_state = (
            await self.tools.get_state()
        )

        # Get raw tree from cache
        raw_tree = self.tools.raw_tree_cache

        return {
            "formatted_text": formatted_text,
            "focused_element": focused_element,
            "ui_elements": ui_elements,
            "phone_state": phone_state,
            "raw_tree": raw_tree,
            "timestamp": datetime.now().isoformat(),
        }
    
    @ensure_connected
    async def get_raw_tree(self) -> Dict[str, Any]:
        """
        Lightweight function to retrieve just the raw accessibility tree state.
        """
        await self.tools.get_state()
        return {"raw_tree": self.tools.raw_tree_cache}

    @ensure_connected
    async def detect_state_change(self) -> UIStateChange | None:
        """
        Check if device state has changed since last check and return the change.

        Returns:
        The UIStateChange object or None if no change

        Use this for continuous monitoring in background tasks.
        NOTE: this is the primary method called by observation tracker to detect state changes and use state.
        TODO: This should reflect using canonical UI State DTOs.
        """
        # fetches the raw UI state
        current_state_raw = await self.tools.get_state()
        # use helper to format raw state into DTO
        current_state = self._format_raw_ui_state(current_state_raw)

        # returns None if no change
        # 1) no previous state
        # 2) state change classified to be NO_CHANGE

        # case 1)
        if self.last_state is None:
            self.last_state = current_state
            return None

        # case 2)
        # NOTE: change types are only APP_SWITCH, CHANGED, or NO_CHANGE
        change_type = self._classify_change(self.last_state, current_state)
        if change_type == StateChangeType.NO_CHANGE:
            return None
        
        # otherwise, update the last state and return the change
        change_event = UIStateChange(
            timestamp=datetime.now(),
            change_type=change_type,
            before=self.last_state,
            after=current_state,
            source=StateChangeSource.OBSERVATION,
            description=None # NOTE: to be added w/ LLM summarization, only on core changes or w/ hashed look-up
        )
        self.last_state = current_state

        return change_event

    @ensure_connected
    async def take_screenshot(self, hide_overlay: bool = True) -> bytes:
        """
        Capture device screenshot.

        Args:
            hide_overlay: Hide Portal overlay in screenshot (default: True)

        Returns:
            Screenshot bytes (PNG format)
        """
        _, screenshot_bytes = await self.tools.take_screenshot(hide_overlay=hide_overlay)
        return screenshot_bytes

    @ensure_connected
    async def get_installed_apps(self, include_system: bool = False) -> List[Dict[str, str]]:
        """
        Get list of installed apps.

        Args:
            include_system: Include system apps (default: False)

        Returns:
            List of dicts with 'package' and 'name' keys
        """
        apps = await self.tools.get_apps(include_system=include_system)
        return apps

    # =====================================================================
    # DIRECT ACTIONS (for fine-grained control)
    # =====================================================================

    @ensure_connected
    async def tap_by_index(self, index: int) -> str:
        """
        Tap UI element by index number.

        Args:
            index: Element index from get_current_state()['ui_elements']

        Returns:
            Result message
        """
        return await self.tools.tap_by_index(index)

    @ensure_connected
    async def input_text(self, text: str, index: int = -1, clear: bool = False) -> str:
        """
        Input text into focused or specified element.

        Args:
            text: Text to input
            index: Element index (-1 for currently focused)
            clear: Clear existing text first

        Returns:
            Result message
        """
        return await self.tools.input_text(text, index=index, clear=clear)

    @ensure_connected
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 500,
    ) -> bool:
        """
        Perform swipe gesture.

        Args:
            start_x, start_y: Start coordinates
            end_x, end_y: End coordinates
            duration_ms: Swipe duration in milliseconds

        Returns:
            Result message
        """
        return await self.tools.swipe(start_x, start_y, end_x, end_y, duration_ms)

    @ensure_connected
    async def back(self) -> str:
        """Press back button."""
        return await self.tools.back()

    @ensure_connected
    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        """
        Launch app by package name.

        Args:
            package: App package name (e.g., 'com.android.settings')
            activity: Specific activity to launch (optional)

        Returns:
            Result message
        """

        return await self.tools.start_app(package, activity)

    # =====================================================================
    # EXECUTION HISTORY (for memory updates)
    # =====================================================================

    def get_execution_history(
        self,
        limit: Optional[int] = None,
        notable_only: bool = False,
    ) -> list[ExecutionResult]:
        """
        Get recent execution history for memory updates.

        Args:
            limit: Max number of executions to return (None = all)
            notable_only: Only return notable changes (app switches, screen changes)

        Returns:
            List of ExecutionResult DTOs
        """
        results = self.execution_history

        if notable_only:
            notable_types = {StateChangeType.APP_SWITCH, StateChangeType.CHANGED}
            results = [r for r in results if r.change_type in notable_types]

        if limit:
            results = results[-limit:]

        return results

    def clear_execution_history(self) -> None:
        """Clear execution history (useful after persisting to database)."""
        self.execution_history = []

    # =====================================================================
    # HELPER METHODS
    # =====================================================================
    
    def _format_raw_ui_state(self, raw_state: Tuple) -> UIState:
        """
        Takes a raw state tuple and formats it into a string.
        - Hashes the state for efficient storage (TBD)
        Args:
            raw_state: Raw state tuple
        Returns:
            Formatted UI state DTO
        """
        raw_tree = self.tools.raw_tree_cache
        # TODO: make a hash w/ this tree
        # TODO: make a hash of the raw UI state, check history to fetch state_id if exists or make new.
        # for now, just make a new state_id for each UI state, regardless of duplicates.

        state_id = str(uuid.uuid4())

        # parse focused_element safely - handle non-numeric values
        focused_element = None
        if raw_state[1]:
            try:
                focused_element = int(raw_state[1])
            except (ValueError, TypeError):
                # if focused_element is non-numeric (e.g., "YouTube"), set to None
                focused_element = None

        current_state = UIState(
            state_id=state_id,
            package=raw_state[3]["packageName"],
            activity=UIActivity(activity=raw_state[3].get("activityName", "unknown")),
            ui_elements=raw_state[2],
            focused_element=focused_element,
            raw_tree=raw_tree,
        )

        return current_state

    def _classify_change(self, before: UIState, after: UIState) -> StateChangeType:
        """
        NOTE: Classifies only whether a state diff is notable or not.
        - The only deterministic state change we can filter for is app switch.
        - The rest are processed downstream as natural language by the observation inferencer.
        - APP_SWITCH is appened to the history as a clear, intermediate signal for switch.

        Returns:
            An enum for the change type:
            'changed', 'no_change', 'app_switch'
        """

        before_pkg = before.package
        after_pkg = after.package

        # case 1) app switch
        if before_pkg != after_pkg:
            return StateChangeType.APP_SWITCH
        
        # case 2) there is no meaningful change
        # NOTE: we're being very narrow about NO_CHANGE condition to ensure we try to capture as many changes as possible
        # TBD, may need more aggresive filtering to reduce observation inference tokens.
        if (
            before_pkg == after_pkg and \
            before.activity == after.activity and \
            before.focused_element == after.focused_element
        ):
            return StateChangeType.NO_CHANGE

        # case 3) otherwise, assume there is a meaningful change
        return StateChangeType.CHANGED
