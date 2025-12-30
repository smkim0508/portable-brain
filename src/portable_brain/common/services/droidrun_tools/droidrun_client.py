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

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# DroidRun SDK imports
from droidrun import DroidAgent, AdbTools, DroidrunConfig, DeviceConfig, AgentConfig
from droidrun.agent import ResultEvent
from droidrun import load_llm

# settings
from portable_brain.config.app_config import get_service_settings

from portable_brain.common.logging.logger import logger

class DroidRunClient:
    """
    Wrapper for DroidRun SDK to integrate with FastAPI memory agent service.

    Provides two interfaces:
    1. High-level: Execute enriched natural language commands via DroidAgent
    2. Low-level: Monitor device state and actions for memory updates
    """

    def __init__(
        self,
        device_serial: str = "emulator-5554",
        use_tcp: bool = True,
        llm_instance=None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize DroidRun client.

        Args:
            device_serial: Android device serial number
            use_tcp: Use TCP mode for faster Portal communication (default: True)
            llm_instance: LLM instance for DroidAgent (if None, will use load_llm)
            api_key: Google API key (if using load_llm, otherwise reads from env)
        """
        self.device_serial = device_serial
        self.use_tcp = use_tcp

        # Initialize LLM: use provided instance or load from DroidRun
        if llm_instance is not None:
            self.llm = llm_instance
        else:
            # Use DroidRun's load_llm with Google provider
            # Set API key in environment if provided
            # NOTE: THIS IS A TEST! seeing if load_llm() can correctly pick out GOOGLE_API_KEY if set, without explicitly passing it as an argument.
            if api_key:
                import os
                os.environ['GOOGLE_API_KEY'] = api_key

            self.llm = load_llm(
                provider_name="GoogleGenAI",
                model="gemini-2.5-flash-lite"
            )

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
        self.last_state: Optional[Tuple] = None
        self.action_history: List[Dict[str, Any]] = []

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
            state = await self.tools.get_state()
            self.last_state = state
            self._connected = True

            logger.info(f"Connected to device {self.device_serial}")
            logger.info(f"Current app: {state[3].get('packageName', 'Unknown')}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to device: {e}")
            self._connected = False
            return False

    # =====================================================================
    # HIGH-LEVEL INTERFACE: Execute Enriched Commands
    # =====================================================================

    async def execute_command(
        self,
        enriched_command: str,
        reasoning: bool = False,
        timeout: int = 120,
    ) -> ResultEvent:
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
            ResultEvent with:
                - success: bool
                - reason: str (explanation or answer)
                - steps: int (number of steps taken)
                - structured_output: Optional[BaseModel] (if output_model was provided)

        Example:
            result = await client.execute_command(
                "Open Settings and tell me the Android version"
            )
            print(f"Success: {result.success}")
            print(f"Answer: {result.reason}")
        """
        if not self.llm:
            raise ValueError("LLM instance required for execute_command()")

        if not self._connected:
            await self.connect()

        # Capture state before execution
        state_before = await self.tools.get_state()

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
        state_after = await self.tools.get_state()

        # Record action for memory agent
        action_record = {
            "timestamp": datetime.now().isoformat(),
            "command": enriched_command,
            "success": result.success,
            "reason": result.reason,
            "steps": result.steps,
            "state_before": self._serialize_state(state_before),
            "state_after": self._serialize_state(state_after),
            "change_type": self._classify_change(state_before, state_after),
        }

        self.action_history.append(action_record)
        self.last_state = state_after

        return result

    # =====================================================================
    # LOW-LEVEL INTERFACE: Monitor Device State
    # =====================================================================

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
        if not self._connected:
            await self.connect()

        # Get processed state
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

    async def detect_state_change(self) -> Optional[Dict[str, Any]]:
        """
        Check if device state has changed since last check.

        Returns:
            None if no change, otherwise dict with:
                - change_type: str ('app_switch', 'screen_change', 'layout_change', 'minor')
                - before: Dict (previous state summary)
                - after: Dict (current state summary)
                - timestamp: str (ISO format)

        Use this for continuous monitoring in background tasks.
        """
        current_state = await self.tools.get_state()

        if self.last_state is None:
            self.last_state = current_state
            return None

        change_type = self._classify_change(self.last_state, current_state)

        if change_type == "no_change":
            return None

        change_event = {
            "change_type": change_type,
            "before": self._serialize_state(self.last_state),
            "after": self._serialize_state(current_state),
            "timestamp": datetime.now().isoformat(),
        }

        self.last_state = current_state
        return change_event

    async def take_screenshot(self, hide_overlay: bool = True) -> bytes:
        """
        Capture device screenshot.

        Args:
            hide_overlay: Hide Portal overlay in screenshot (default: True)

        Returns:
            Screenshot bytes (PNG format)
        """
        if not self._connected:
            await self.connect()

        _, screenshot_bytes = await self.tools.take_screenshot(hide_overlay=hide_overlay)
        return screenshot_bytes

    async def get_installed_apps(self, include_system: bool = False) -> List[Dict[str, str]]:
        """
        Get list of installed apps.

        Args:
            include_system: Include system apps (default: False)

        Returns:
            List of dicts with 'package' and 'name' keys
        """
        if not self._connected:
            await self.connect()

        apps = await self.tools.get_apps(include_system=include_system)
        return apps

    # =====================================================================
    # DIRECT ACTIONS (for fine-grained control)
    # =====================================================================

    async def tap_by_index(self, index: int) -> str:
        """
        Tap UI element by index number.

        Args:
            index: Element index from get_current_state()['ui_elements']

        Returns:
            Result message
        """
        if not self._connected:
            await self.connect()

        return await self.tools.tap_by_index(index)

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
        if not self._connected:
            await self.connect()

        return await self.tools.input_text(text, index=index, clear=clear)

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
        if not self._connected:
            await self.connect()

        return await self.tools.swipe(start_x, start_y, end_x, end_y, duration_ms)

    async def back(self) -> str:
        """Press back button."""
        if not self._connected:
            await self.connect()

        return await self.tools.back()

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        """
        Launch app by package name.

        Args:
            package: App package name (e.g., 'com.android.settings')
            activity: Specific activity to launch (optional)

        Returns:
            Result message
        """
        if not self._connected:
            await self.connect()

        return await self.tools.start_app(package, activity)

    # =====================================================================
    # ACTION HISTORY (for memory updates)
    # =====================================================================

    def get_action_history(
        self,
        limit: Optional[int] = None,
        notable_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get recent action history for memory updates.

        Args:
            limit: Max number of actions to return (None = all)
            notable_only: Only return notable changes (app switches, screen changes)

        Returns:
            List of action records with timestamps, commands, state changes

        Example:
            recent_actions = client.get_action_history(limit=10, notable_only=True)
            for action in recent_actions:
                if action['change_type'] in ['app_switch', 'screen_change']:
                    # Update memory graph
                    pass
        """
        actions = self.action_history

        if notable_only:
            actions = [
                a for a in actions
                if a.get("change_type") in ["app_switch", "screen_change", "major_layout_change"]
            ]

        if limit:
            actions = actions[-limit:]

        return actions

    def clear_action_history(self) -> None:
        """Clear action history (useful after persisting to database)."""
        self.action_history = []

    # =====================================================================
    # HELPER METHODS
    # =====================================================================

    def _serialize_state(self, state: Tuple) -> Dict[str, Any]:
        """
        Convert state tuple to serializable dict.

        State tuple format: (formatted_text, focused_element, ui_elements, phone_state)
        """
        if not state or len(state) < 4:
            return {}

        return {
            "package": state[3].get("packageName", ""),
            "activity": state[3].get("currentApp", ""),
            "element_count": len(state[2]),
            "is_editable": state[3].get("isEditable", False),
            "focused": state[1] if state[1] else None,
        }

    def _classify_change(self, before: Tuple, after: Tuple) -> str:
        """
        Classify type of UI change between two states.

        Returns:
            'no_change', 'app_switch', 'screen_change', 'major_layout_change',
            'minor_layout_change', or 'scroll_or_animation'
        """
        if not before or not after or len(before) < 4 or len(after) < 4:
            return "unknown"

        before_pkg = before[3].get("packageName", "")
        after_pkg = after[3].get("packageName", "")

        # No change
        if (
            before_pkg == after_pkg
            and before[3].get("currentApp") == after[3].get("currentApp")
            and len(before[2]) == len(after[2])
        ):
            return "no_change"

        # App switch
        if before_pkg != after_pkg:
            return "app_switch"

        # Screen change (different activity)
        if before[3].get("currentApp") != after[3].get("currentApp"):
            return "screen_change"

        # Element count change
        before_count = len(before[2])
        after_count = len(after[2])
        diff = abs(before_count - after_count)

        if diff > 20:
            return "major_layout_change"
        elif diff > 5:
            return "minor_layout_change"
        else:
            return "scroll_or_animation"


# =====================================================================
# USAGE EXAMPLES
# =====================================================================

async def example_high_level_execution():
    """Example: Execute enriched commands from memory agent."""
    from droidrun import load_llm

    GOOGLE_API_KEY = get_service_settings().GOOGLE_GENAI_API_KEY

    # Load LLM (same as your memory agent uses)
    llm = load_llm(provider_name="GoogleGenAI", model="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY)

    # Initialize client
    client = DroidRunClient(
        device_serial="emulator-5554",
        use_tcp=True,
        llm_instance=llm,
    )

    # Connect to device
    await client.connect()

    # Execute enriched command (from your memory agent)
    result = await client.execute_command(
        enriched_command="Open Messages app, send SMS to Kevin Chen (+1-234-567-8900) with message 'about dinner'",
        reasoning=True,  # Enable planning for complex tasks
    )

    print(f"Success: {result.success}")
    print(f"Explanation: {result.reason}")
    print(f"Steps taken: {result.steps}")

    # Get action history for memory updates
    actions = client.get_action_history(notable_only=True)
    print(f"Notable actions: {len(actions)}")


async def example_low_level_monitoring():
    """Example: Monitor device state for memory updates."""
    client = DroidRunClient(device_serial="emulator-5554")
    await client.connect()

    # Continuous monitoring loop (for background task)
    while True:
        change = await client.detect_state_change()

        if change:
            print(f"Detected: {change['change_type']}")
            print(f"From: {change['before']['package']}")
            print(f"To: {change['after']['package']}")

            # Update memory graph based on change
            if change['change_type'] in ['app_switch', 'screen_change']:
                # Send to memory handler
                pass

        await asyncio.sleep(1)  # Poll every second


async def example_direct_actions():
    """Example: Direct device control."""
    client = DroidRunClient(device_serial="emulator-5554")
    await client.connect()

    # Get current state
    state = await client.get_current_state()
    print(f"Current app: {state['phone_state']['packageName']}")
    print(f"UI elements: {len(state['ui_elements'])}")

    # Find and tap a button
    for idx, element in enumerate(state['ui_elements']):
        if element.get('text') == 'Settings':
            await client.tap_by_index(idx)
            break

    # Take screenshot
    screenshot = await client.take_screenshot()
    with open('screenshot.png', 'wb') as f:
        f.write(screenshot)


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_high_level_execution())