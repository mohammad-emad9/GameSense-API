"""Counter-Strike 2 game adapter.

This adapter provides telemetry collection for Counter-Strike 2 (CS2)
using Game State Integration (GSI) logs and window observation.

Example:
    >>> from adapters.games.cs2 import CS2Adapter
    >>> adapter = CS2Adapter()
    >>> if adapter.is_game_running():
    ...     context = await adapter.get_context()
    ...     print(f"Map: {context.metadata.get('map')}")
"""

from __future__ import annotations

import json
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


class CS2GSIStrategy(LogReaderStrategy):
    """Specialized log reader for CS2 Game State Integration.

    CS2 can output game state via GSI to a local file or HTTP endpoint.
    This strategy reads local GSI output files.

    The GSI config should be placed in:
    Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/cfg/gamestate_integration_gamesense.cfg
    """

    def __init__(self, gsi_path: Optional[Path] = None) -> None:
        """Initialize CS2 GSI strategy.

        Args:
            gsi_path: Path to GSI output file. If None, uses default location.
        """
        # Try to find GSI log path
        if gsi_path is None:
            gsi_path = self._find_gsi_path()

        config = {
            "log_path": str(gsi_path) if gsi_path else "",
            "format": "json",
            "tail_lines": 10,
        }

        super().__init__(config)
        self.priority = 1  # Highest priority for CS2

    def _find_gsi_path(self) -> Optional[Path]:
        """Try to find GSI output file location.

        Returns:
            Path to GSI file, or None if not found.
        """
        # Common Steam installation paths
        steam_paths = [
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Steam",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Steam",
            Path("D:/Steam"),
            Path("E:/Steam"),
            Path("D:/SteamLibrary"),
        ]

        for steam_path in steam_paths:
            gsi_file = (
                steam_path
                / "steamapps/common/Counter-Strike Global Offensive/game/csgo/gsi_output.json"
            )
            if gsi_file.exists():
                return gsi_file

        return None

    async def read(self) -> Dict[str, Any]:
        """Read CS2 GSI data.

        Returns:
            Parsed GSI data with state, stats, and metadata.
        """
        raw_data = await super().read()

        # Transform GSI-specific fields to standard format
        result: Dict[str, Any] = {
            "state": {},
            "stats": {},
            "metadata": {},
        }

        # Extract player stats
        if "player" in raw_data.get("state", {}):
            player = raw_data["state"]["player"]
            if isinstance(player, dict):
                result["stats"]["kills"] = player.get("kills", 0)
                result["stats"]["deaths"] = player.get("deaths", 0)
                result["stats"]["assists"] = player.get("assists", 0)
                result["stats"]["health"] = player.get("health", 100)
                result["stats"]["armor"] = player.get("armor", 0)
                result["stats"]["money"] = player.get("money", 0)

        # Extract round info
        if "round" in raw_data.get("state", {}):
            round_info = raw_data["state"]["round"]
            if isinstance(round_info, dict):
                result["state"]["round_phase"] = round_info.get("phase", "unknown")
                result["state"]["bomb_planted"] = round_info.get("bomb", "") == "planted"

        # Extract map info
        if "map" in raw_data.get("state", {}):
            map_info = raw_data["state"]["map"]
            if isinstance(map_info, dict):
                result["metadata"]["map"] = map_info.get("name", "unknown")
                result["metadata"]["mode"] = map_info.get("mode", "unknown")
                result["state"]["round_number"] = map_info.get("round", 0)
                result["stats"]["team_ct_score"] = map_info.get("team_ct", {}).get("score", 0)
                result["stats"]["team_t_score"] = map_info.get("team_t", {}).get("score", 0)

        # Also merge any direct stats from raw data
        result["stats"].update(raw_data.get("stats", {}))
        result["state"].update({k: v for k, v in raw_data.get("state", {}).items() if k not in ["player", "round", "map"]})

        return result


class CS2WindowStrategy(WindowObserverStrategy):
    """Specialized window observer for CS2.

    Extracts game information from CS2 window title and state.
    """

    def __init__(self) -> None:
        """Initialize CS2 window strategy."""
        config = {
            "process_name": "cs2.exe",
            "title_patterns": {
                # CS2 window title patterns
                "map_from_title": r"Counter-Strike 2 - (\w+)",
            },
        }
        super().__init__(config)
        self.priority = 2


class CS2Adapter(BaseGameAdapter):
    """Game adapter for Counter-Strike 2.

    Supports telemetry collection via:
    1. Game State Integration (GSI) - Recommended
    2. Window title observation - Fallback

    Attributes:
        game_id: "cs2"
        display_name: "Counter-Strike 2"
        process_names: ["cs2.exe"]

    Example:
        >>> adapter = CS2Adapter()
        >>> if adapter.is_game_running():
        ...     ctx = await adapter.get_context()
        ...     print(f"Playing on {ctx.metadata.get('map')}")
    """

    game_id = "cs2"
    display_name = "Counter-Strike 2"
    process_names = ["cs2.exe"]
    version = "1.0.0"
    author = "GameSense Team"
    description = "Counter-Strike 2 adapter with GSI and window observation support"

    def _init_strategies(self) -> None:
        """Initialize CS2 telemetry strategies."""
        # Add GSI strategy (highest priority)
        self.add_strategy(CS2GSIStrategy())

        # Add window observer as fallback
        self.add_strategy(CS2WindowStrategy())

        logger.info(
            f"CS2 adapter initialized with {len(self._strategies)} strategies",
            extra={"component": "adapter", "game_id": self.game_id},
        )
