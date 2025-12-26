"""Base classes for game adapters and telemetry strategies.

This module defines the abstract base classes that all game adapters
and telemetry strategies must implement.

Example:
    >>> from adapters.base import BaseGameAdapter, BaseTelemetryStrategy
    >>> class MyGameAdapter(BaseGameAdapter):
    ...     game_id = "mygame"
    ...     display_name = "My Game"
    ...     process_names = ["mygame.exe"]
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import psutil

from models.game import AdapterInfo, GameContext, TelemetrySource

# Configure logging
logger = logging.getLogger(__name__)


class BaseTelemetryStrategy(ABC):
    """Abstract base class for telemetry data collection strategies.

    A strategy encapsulates a single method of collecting game data,
    such as parsing log files, reading window properties, or observing
    network packets.

    Attributes:
        source_type: The type of telemetry source this strategy uses.
        priority: Lower values = higher priority (tried first).
        requires_admin: Whether this strategy requires elevated privileges.

    Example:
        >>> class LogReaderStrategy(BaseTelemetryStrategy):
        ...     source_type = TelemetrySource.LOG
        ...     priority = 1
        ...     requires_admin = False
    """

    source_type: TelemetrySource = TelemetrySource.LOG
    priority: int = 100
    requires_admin: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the strategy with optional configuration.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self._available: Optional[bool] = None

    @abstractmethod
    async def read(self) -> Dict[str, Any]:
        """Read telemetry data from this source.

        Returns:
            Dictionary containing telemetry data with keys like
            'state', 'stats', 'metadata'.

        Raises:
            Exception: If reading fails (caught by adapter).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy is currently available.

        Returns:
            True if the data source is accessible, False otherwise.
        """
        pass

    def get_source_type(self) -> TelemetrySource:
        """Get the telemetry source type for this strategy.

        Returns:
            The TelemetrySource enum value.
        """
        return self.source_type


class BaseGameAdapter(ABC):
    """Abstract base class for game-specific adapters.

    Each adapter handles telemetry collection for a specific game,
    using one or more telemetry strategies in priority order.

    Attributes:
        game_id: Unique identifier for this game (e.g., "cs2").
        display_name: Human-readable game name.
        process_names: List of executable names to detect this game.
        version: Adapter version string.
        author: Adapter author/maintainer.
        description: Brief description of the adapter.

    Example:
        >>> class CS2Adapter(BaseGameAdapter):
        ...     game_id = "cs2"
        ...     display_name = "Counter-Strike 2"
        ...     process_names = ["cs2.exe"]
    """

    game_id: str = ""
    display_name: str = ""
    process_names: List[str] = []
    version: str = "1.0.0"
    author: str = "GameSense Team"
    description: str = ""

    def __init__(self) -> None:
        """Initialize the adapter and its strategies."""
        self._strategies: List[BaseTelemetryStrategy] = []
        self._context: Optional[GameContext] = None
        self._session_id: Optional[str] = None
        self._last_update: Optional[datetime] = None

        # Initialize strategies
        self._init_strategies()

        logger.info(
            f"Adapter '{self.display_name}' initialized with "
            f"{len(self._strategies)} strategies",
            extra={"component": "adapter", "game_id": self.game_id},
        )

    @abstractmethod
    def _init_strategies(self) -> None:
        """Initialize telemetry strategies for this adapter.

        Subclasses must implement this to add their strategies to
        self._strategies in priority order.
        """
        pass

    def add_strategy(self, strategy: BaseTelemetryStrategy) -> None:
        """Add a telemetry strategy to this adapter.

        Strategies are sorted by priority after adding.

        Args:
            strategy: The strategy instance to add.
        """
        self._strategies.append(strategy)
        self._strategies.sort(key=lambda s: s.priority)

    def is_game_running(self) -> bool:
        """Check if this game is currently running.

        Returns:
            True if any of the game's processes are running.
        """
        try:
            for proc in psutil.process_iter(["name"]):
                proc_name = proc.info.get("name", "").lower()
                if proc_name in [p.lower() for p in self.process_names]:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return False

    def get_process(self) -> Optional[psutil.Process]:
        """Get the game's process object if running.

        Returns:
            The Process object, or None if not running.
        """
        try:
            for proc in psutil.process_iter(["name", "pid"]):
                proc_name = proc.info.get("name", "").lower()
                if proc_name in [p.lower() for p in self.process_names]:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return None

    async def get_context(self) -> GameContext:
        """Get the current game context using available strategies.

        Tries each strategy in priority order, falling back to the next
        if one fails. Returns a minimal context if all strategies fail.

        Returns:
            GameContext with data from available sources.
        """
        # Check if game is running
        is_running = self.is_game_running()

        if not is_running:
            # Return inactive context
            return GameContext(
                game_id=self.game_id,
                active=False,
                state={},
                stats={},
                metadata={"display_name": self.display_name},
                sources_used=[],
            )

        # Initialize or reuse context
        if self._context is None or not self._context.active:
            from uuid import uuid4

            self._session_id = str(uuid4())
            self._context = GameContext(
                game_id=self.game_id,
                active=True,
                session_id=self._session_id,
                metadata={"display_name": self.display_name},
            )

        # Collect data from strategies
        sources_used: List[TelemetrySource] = []
        collected_state: Dict[str, Any] = {}
        collected_stats: Dict[str, Any] = {}
        collected_metadata: Dict[str, Any] = {"display_name": self.display_name}

        for strategy in self._strategies:
            try:
                if not strategy.is_available():
                    continue

                data = await strategy.read()

                # Merge collected data
                if "state" in data:
                    collected_state.update(data["state"])
                if "stats" in data:
                    collected_stats.update(data["stats"])
                if "metadata" in data:
                    collected_metadata.update(data["metadata"])

                sources_used.append(strategy.get_source_type())

                logger.debug(
                    f"Strategy {strategy.source_type.value} provided data",
                    extra={"component": "adapter", "game_id": self.game_id},
                )

            except Exception as e:
                logger.warning(
                    f"Strategy {strategy.source_type.value} failed: {e}",
                    extra={"component": "adapter", "game_id": self.game_id},
                )
                continue

        # Update context
        self._context.active = True
        self._context.state = collected_state
        self._context.stats = collected_stats
        self._context.metadata = collected_metadata
        self._context.sources_used = sources_used
        self._context.timestamp = datetime.utcnow()
        self._last_update = datetime.utcnow()

        return self._context

    def get_info(self) -> AdapterInfo:
        """Get adapter information.

        Returns:
            AdapterInfo with this adapter's details.
        """
        return AdapterInfo(
            id=self.game_id,
            name=self.display_name,
            version=self.version,
            supported_strategies=[s.source_type for s in self._strategies],
            process_names=self.process_names,
            description=self.description,
            author=self.author,
        )

    def reset_session(self) -> None:
        """Reset the current game session.

        Called when the game exits to prepare for next session.
        """
        self._context = None
        self._session_id = None
        self._last_update = None
        logger.info(
            f"Session reset for {self.display_name}",
            extra={"component": "adapter", "game_id": self.game_id},
        )
