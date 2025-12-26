"""Window observer telemetry strategy.

This strategy observes window titles and properties to extract game state.
Works with any game without requiring log files or memory access.

Example:
    >>> strategy = WindowObserverStrategy(config={
    ...     "process_name": "cs2.exe",
    ...     "title_patterns": {"map": r"- (\w+)$"}
    ... })
    >>> if strategy.is_available():
    ...     data = await strategy.read()
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import psutil

from adapters.base import BaseTelemetryStrategy
from models.game import TelemetrySource

# Windows API constants
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Configure logging
logger = logging.getLogger(__name__)


class WindowObserverStrategy(BaseTelemetryStrategy):
    """Telemetry strategy that observes game window properties.

    Extracts information from window titles and properties. This is
    a fallback strategy when log files aren't available.

    Attributes:
        source_type: TelemetrySource.WINDOW
        priority: 2 (lower than log reader)
        requires_admin: False

    Example:
        >>> strategy = WindowObserverStrategy(config={
        ...     "process_name": "GTA5.exe",
        ...     "title_patterns": {"mode": r"(Online|Story)"}
        ... })
    """

    source_type = TelemetrySource.WINDOW
    priority = 2
    requires_admin = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the window observer strategy.

        Args:
            config: Configuration dict with keys:
                - process_name: Target process name (required)
                - title_patterns: Dict of name -> regex for title parsing
                - window_class: Optional window class name filter
        """
        super().__init__(config)

        self.process_name: str = config.get("process_name", "") if config else ""
        self.title_patterns: Dict[str, str] = config.get("title_patterns", {}) if config else {}
        self.window_class: Optional[str] = config.get("window_class") if config else None
        self._cached_hwnd: Optional[int] = None
        self._cached_pid: Optional[int] = None

    def is_available(self) -> bool:
        """Check if the target window is available.

        Returns:
            True if window is found, False otherwise.
        """
        if not self.process_name:
            return False

        hwnd = self._find_window()
        return hwnd is not None and hwnd != 0

    async def read(self) -> Dict[str, Any]:
        """Read telemetry data from window properties.

        Returns:
            Dictionary with 'state' and 'metadata' from window info.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_window)

    def _read_window(self) -> Dict[str, Any]:
        """Read window data (blocking).

        Returns:
            Dictionary with extracted window data.
        """
        result: Dict[str, Any] = {
            "state": {},
            "stats": {},
            "metadata": {},
        }

        hwnd = self._find_window()
        if not hwnd:
            return result

        # Get window title
        title = self._get_window_title(hwnd)
        if title:
            result["metadata"]["window_title"] = title

            # Apply title patterns
            for key, pattern in self.title_patterns.items():
                try:
                    match = re.search(pattern, title)
                    if match:
                        value = match.group(1) if match.groups() else match.group(0)
                        # Try to convert to number
                        try:
                            if "." in value:
                                result["stats"][key] = float(value)
                            else:
                                result["stats"][key] = int(value)
                        except ValueError:
                            result["state"][key] = value
                except re.error as e:
                    logger.warning(
                        f"Invalid regex pattern for {key}: {e}",
                        extra={"component": "strategy"},
                    )

        # Get window state
        window_state = self._get_window_state(hwnd)
        result["state"].update(window_state)

        return result

    def _find_window(self) -> Optional[int]:
        """Find the game window handle.

        Returns:
            Window handle (HWND) or None if not found.
        """
        # First, find the process
        target_pid = None
        try:
            for proc in psutil.process_iter(["name", "pid"]):
                if proc.info["name"] and proc.info["name"].lower() == self.process_name.lower():
                    target_pid = proc.info["pid"]
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        if not target_pid:
            return None

        # Cache for efficiency
        self._cached_pid = target_pid

        # Enumerate windows to find one owned by this process
        result: List[int] = []

        def enum_windows_callback(hwnd: int, _: Any) -> bool:
            """Callback for EnumWindows."""
            try:
                # Get window's process ID
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                # Check if it belongs to target process
                if pid.value == target_pid:
                    # Check if window is visible
                    if user32.IsWindowVisible(hwnd):
                        # Optional: check window class
                        if self.window_class:
                            class_name = ctypes.create_unicode_buffer(256)
                            user32.GetClassNameW(hwnd, class_name, 256)
                            if class_name.value != self.window_class:
                                return True  # Continue enumeration

                        result.append(hwnd)
            except Exception:
                pass
            return True  # Continue enumeration

        # Define callback type
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        callback = WNDENUMPROC(enum_windows_callback)

        user32.EnumWindows(callback, 0)

        if result:
            self._cached_hwnd = result[0]
            return result[0]

        return None

    def _get_window_title(self, hwnd: int) -> str:
        """Get the window title.

        Args:
            hwnd: Window handle.

        Returns:
            Window title string.
        """
        try:
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                return buffer.value
        except Exception as e:
            logger.debug(f"Failed to get window title: {e}")

        return ""

    def _get_window_state(self, hwnd: int) -> Dict[str, Any]:
        """Get window state information.

        Args:
            hwnd: Window handle.

        Returns:
            Dictionary with window state.
        """
        state: Dict[str, Any] = {}

        try:
            # Check if window is focused
            foreground = user32.GetForegroundWindow()
            state["is_focused"] = foreground == hwnd

            # Check if minimized
            state["is_minimized"] = bool(user32.IsIconic(hwnd))

            # Check if maximized
            state["is_maximized"] = bool(user32.IsZoomed(hwnd))

            # Get window rectangle for screen position
            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                ]

            rect = RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                state["window_width"] = rect.right - rect.left
                state["window_height"] = rect.bottom - rect.top

        except Exception as e:
            logger.debug(f"Failed to get window state: {e}")

        return state
