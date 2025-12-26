"""Universal game context models for GameSense API.

This module defines game-agnostic data models for telemetry collection
from any Windows game. All game-specific logic lives in adapter plugins.

Example:
    >>> from models.game import GameContext
    >>> context = GameContext(
    ...     game_id="cs2",
    ...     active=True,
    ...     state={"round_phase": "live", "map": "de_dust2"},
    ...     stats={"kills": 5, "deaths": 2, "ping": 42}
    ... )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TelemetrySource(str, Enum):
    """Enumeration of telemetry data sources.

    Ordered by priority (lowest risk/admin requirement first).

    Attributes:
        LOG: Game log file parsing (no admin required).
        WINDOW: Window title/properties observation (no admin required).
        NETWORK: UDP/TCP telemetry packets (no admin required).
        MEMORY: Process memory reading (admin required, opt-in only).
    """

    LOG = "log"
    WINDOW = "window"
    NETWORK = "network"
    MEMORY = "memory"


class GameContext(BaseModel):
    """Universal game context model.

    A game-agnostic container for telemetry data collected from any
    Windows game. All game-specific interpretation happens in adapters.

    Attributes:
        game_id: Unique identifier for the game (e.g., "cs2", "gta5").
        active: Whether the game is currently running and in focus.
        timestamp: When this context was last updated (UTC).
        session_id: Unique identifier for this game session.
        state: Dynamic game state data (e.g., round phase, menu state).
        stats: Player/game statistics (e.g., kills, ping, score).
        metadata: Static game info (e.g., map name, version, mode).
        sources_used: Which telemetry sources provided this data.

    Example:
        >>> ctx = GameContext(
        ...     game_id="cs2",
        ...     active=True,
        ...     state={"round_phase": "live", "bomb_planted": True},
        ...     stats={"kills": 12, "deaths": 5, "ping": 35},
        ...     metadata={"map": "de_inferno", "mode": "competitive"}
        ... )
    """

    game_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique game identifier (e.g., 'cs2', 'gta5')",
    )
    active: bool = Field(
        default=False,
        description="Whether the game is currently running",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this context was last updated (UTC)",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this game session",
    )
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic game state (e.g., round_phase, menu_state)",
    )
    stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Player/game statistics (e.g., kills, ping, score)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Static game info (e.g., map, version, mode)",
    )
    sources_used: List[TelemetrySource] = Field(
        default_factory=list,
        description="Telemetry sources that provided this data",
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "game_id": "cs2",
                "active": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "state": {
                    "round_phase": "live",
                    "bomb_planted": False,
                    "freeze_time": False,
                },
                "stats": {
                    "kills": 12,
                    "deaths": 5,
                    "assists": 3,
                    "ping": 35,
                },
                "metadata": {
                    "map": "de_inferno",
                    "mode": "competitive",
                    "team": "CT",
                },
                "sources_used": ["log", "window"],
            }
        }

    def merge_state(self, new_state: Dict[str, Any]) -> None:
        """Merge new state data into existing state.

        Args:
            new_state: Dictionary of state updates to merge.
        """
        self.state.update(new_state)
        self.timestamp = datetime.utcnow()

    def merge_stats(self, new_stats: Dict[str, Any]) -> None:
        """Merge new stats data into existing stats.

        Args:
            new_stats: Dictionary of stats updates to merge.
        """
        self.stats.update(new_stats)
        self.timestamp = datetime.utcnow()


class AdapterInfo(BaseModel):
    """Information about a registered game adapter.

    Attributes:
        id: Unique adapter identifier (matches game_id).
        name: Human-readable display name.
        version: Adapter version string.
        supported_strategies: List of telemetry sources this adapter supports.
        process_names: Executable names to detect this game.
        description: Brief description of the adapter.
        author: Adapter author/maintainer.

    Example:
        >>> info = AdapterInfo(
        ...     id="cs2",
        ...     name="Counter-Strike 2",
        ...     version="1.0.0",
        ...     supported_strategies=[TelemetrySource.LOG, TelemetrySource.WINDOW],
        ...     process_names=["cs2.exe"]
        ... )
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique adapter identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable display name",
    )
    version: str = Field(
        default="1.0.0",
        description="Adapter version string",
    )
    supported_strategies: List[TelemetrySource] = Field(
        default_factory=list,
        description="Telemetry sources supported by this adapter",
    )
    process_names: List[str] = Field(
        default_factory=list,
        description="Executable names to detect this game",
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the adapter",
    )
    author: Optional[str] = Field(
        default=None,
        description="Adapter author/maintainer",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "id": "cs2",
                "name": "Counter-Strike 2",
                "version": "1.0.0",
                "supported_strategies": ["log", "window"],
                "process_names": ["cs2.exe"],
                "description": "Adapter for Counter-Strike 2 using GSI and window detection",
                "author": "GameSense Team",
            }
        }


class ActiveGameResponse(BaseModel):
    """Response model for active game endpoint.

    Attributes:
        detected: Whether any game is currently detected.
        game: The active game context, if any.
        available_adapters: Number of registered adapters.

    Example:
        >>> response = ActiveGameResponse(
        ...     detected=True,
        ...     game=GameContext(game_id="cs2", active=True)
        ... )
    """

    detected: bool = Field(
        default=False,
        description="Whether any game is currently detected",
    )
    game: Optional[GameContext] = Field(
        default=None,
        description="Active game context, if any",
    )
    available_adapters: int = Field(
        default=0,
        description="Number of registered adapters",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "detected": True,
                "game": {
                    "game_id": "cs2",
                    "active": True,
                    "state": {"round_phase": "live"},
                    "stats": {"kills": 5},
                },
                "available_adapters": 2,
            }
        }
