"""Grand Theft Auto V game adapter.

This adapter provides telemetry collection for GTA V using
window observation and log file reading.

Example:
    >>> from adapters.games.gta5 import GTA5Adapter
    >>> adapter = GTA5Adapter()
    >>> if adapter.is_game_running():
    ...     context = await adapter.get_context()
    ...     print(f"Mode: {context.state.get('mode')}")
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters.base import BaseGameAdapter, BaseTelemetryStrategy
from adapters.strategies.log_reader import LogReaderStrategy
from adapters.strategies.window_observer import WindowObserverStrategy
from models.game import TelemetrySource

# Configure logging
logger = logging.getLogger(__name__)


class GTA5LogStrategy(LogReaderStrategy):
    """Log reader strategy for GTA V launcher logs.

    Reads Rockstar Games Launcher logs to detect game state.
    """

    def __init__(self) -> None:
        """Initialize GTA V log strategy."""
        log_path = self._find_log_path()

        config = {
            "log_path": str(log_path) if log_path else "",
            "format": "custom",
            "tail_lines": 100,
            "patterns": {
                "mode": r"(Online|Story Mode)",
                "loading": r"(Loading|Initializing)",
                "session": r"Joining (.*?) session",
            },
        }

        super().__init__(config)
        self.priority = 1

    def _find_log_path(self) -> Optional[Path]:
        """Find GTA V/Rockstar launcher log file.

        Returns:
            Path to log file, or None if not found.
        """
        # Rockstar Games Launcher log location
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            launcher_log = Path(local_app_data) / "Rockstar Games/Launcher/launcher.log"
            if launcher_log.exists():
                return launcher_log

        # Documents location for GTA V logs
        documents = Path(os.environ.get("USERPROFILE", "")) / "Documents"
        gta_settings = documents / "Rockstar Games/GTA V/settings.xml"
        if gta_settings.exists():
            # Return parent for potential log files
            return gta_settings.parent / "launcher.log"

        return None


class GTA5WindowStrategy(WindowObserverStrategy):
    """Window observer strategy for GTA V.

    Extracts game mode and state from GTA V window title.
    """

    def __init__(self) -> None:
        """Initialize GTA V window strategy."""
        config = {
            "process_name": "GTA5.exe",
            "title_patterns": {
                # GTA V window title patterns
                "game_detected": r"Grand Theft Auto V",
            },
        }
        super().__init__(config)
        self.priority = 2

        # Also check for PlayGTAV.exe (launcher)
        self._alt_process = "PlayGTAV.exe"

    def is_available(self) -> bool:
        """Check if GTA V window is available.

        Returns:
            True if either GTA5.exe or PlayGTAV.exe window is found.
        """
        if super().is_available():
            return True

        # Try alternate process
        original = self.process_name
        self.process_name = self._alt_process
        result = super().is_available()
        self.process_name = original
        return result


class GTA5Adapter(BaseGameAdapter):
    """Game adapter for Grand Theft Auto V.

    Supports telemetry collection via:
    1. Rockstar Launcher logs
    2. Window title observation

    Attributes:
        game_id: "gta5"
        display_name: "Grand Theft Auto V"
        process_names: ["GTA5.exe", "PlayGTAV.exe"]

    Example:
        >>> adapter = GTA5Adapter()
        >>> if adapter.is_game_running():
        ...     ctx = await adapter.get_context()
        ...     print(f"GTA V is running: {ctx.active}")
    """

    game_id = "gta5"
    display_name = "Grand Theft Auto V"
    process_names = ["GTA5.exe", "PlayGTAV.exe"]
    version = "1.0.0"
    author = "GameSense Team"
    description = "GTA V adapter with launcher log and window observation support"

    def _init_strategies(self) -> None:
        """Initialize GTA V telemetry strategies."""
        # Add log reader strategy
        self.add_strategy(GTA5LogStrategy())

        # Add window observer as fallback
        self.add_strategy(GTA5WindowStrategy())

        logger.info(
            f"GTA V adapter initialized with {len(self._strategies)} strategies",
            extra={"component": "adapter", "game_id": self.game_id},
        )

    async def get_context(self) -> Any:
        """Get GTA V game context with additional GTA-specific logic.

        Returns:
            GameContext with GTA V telemetry data.
        """
        context = await super().get_context()

        # Add GTA-specific metadata
        if context.active:
            # Detect if Online or Story mode based on window focus patterns
            if "game_detected" in context.state:
                context.metadata["launcher"] = "Rockstar Games Launcher"

        return context
