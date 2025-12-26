"""Game detection service for GameSense API.

This module provides active game detection by scanning running processes
against registered adapters.

Example:
    >>> from core.game_detector import GameDetector
    >>> from adapters.registry import AdapterRegistry
    >>> detector = GameDetector(registry)
    >>> active_game = detector.detect_active_game()
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

from adapters.base import BaseGameAdapter
from adapters.registry import AdapterRegistry
from models.game import GameContext

# Configure logging
logger = logging.getLogger(__name__)


class GameDetector:
    """Service for detecting active games.

    Scans running processes to identify games with registered adapters.
    Supports event callbacks for game start/stop events.

    Attributes:
        _registry: The adapter registry to check against.
        _active_games: Set of currently detected game IDs.
        _callbacks: Dict of event name -> list of callback functions.

    Example:
        >>> detector = GameDetector(registry)
        >>> detector.on_game_start(lambda game_id: print(f"Started: {game_id}"))
        >>> await detector.start_monitoring()
    """

    def __init__(self, registry: AdapterRegistry) -> None:
        """Initialize the game detector.

        Args:
            registry: The adapter registry to use for detection.
        """
        self._registry = registry
        self._active_games: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = {
            "game_start": [],
            "game_stop": [],
        }
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown: bool = False
        self._lock = asyncio.Lock()

        logger.info(
            "GameDetector initialized",
            extra={"component": "detector"},
        )

    def on_game_start(self, callback: Callable[[str], None]) -> None:
        """Register a callback for game start events.

        Args:
            callback: Function to call with game_id when a game starts.
        """
        self._callbacks["game_start"].append(callback)

    def on_game_stop(self, callback: Callable[[str], None]) -> None:
        """Register a callback for game stop events.

        Args:
            callback: Function to call with game_id when a game stops.
        """
        self._callbacks["game_stop"].append(callback)

    def _emit(self, event: str, game_id: str) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event: Event name ("game_start" or "game_stop").
            game_id: The game identifier.
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(game_id)
            except Exception as e:
                logger.error(
                    f"Error in {event} callback: {e}",
                    extra={"component": "detector", "game_id": game_id},
                )

    def detect_active_games(self) -> List[str]:
        """Detect all currently running games.

        Scans running processes against all registered adapters.

        Returns:
            List of game_ids for detected running games.
        """
        running_games: List[str] = []

        for adapter in self._registry.get_all():
            if adapter.is_game_running():
                running_games.append(adapter.game_id)

        return running_games

    def get_active_adapter(self) -> Optional[BaseGameAdapter]:
        """Get the adapter for the currently active game.

        If multiple games are running, returns the first one found.

        Returns:
            The active game's adapter, or None if no game is running.
        """
        running = self._registry.get_running_games()
        return running[0] if running else None

    async def check_and_update(self) -> Dict[str, Any]:
        """Check for game state changes and emit events.

        Returns:
            Dict with 'started' and 'stopped' lists of game_ids.
        """
        current_games = set(self.detect_active_games())

        started: List[str] = []
        stopped: List[str] = []

        async with self._lock:
            # Check for newly started games
            for game_id in current_games - self._active_games:
                started.append(game_id)
                self._emit("game_start", game_id)
                logger.info(
                    f"Game started: {game_id}",
                    extra={"component": "detector", "game_id": game_id},
                )

            # Check for stopped games
            for game_id in self._active_games - current_games:
                stopped.append(game_id)
                self._emit("game_stop", game_id)

                # Reset adapter session
                adapter = self._registry.get(game_id)
                if adapter:
                    adapter.reset_session()

                logger.info(
                    f"Game stopped: {game_id}",
                    extra={"component": "detector", "game_id": game_id},
                )

            self._active_games = current_games

        return {"started": started, "stopped": stopped}

    async def start_monitoring(self, interval: float = 2.0) -> asyncio.Task:
        """Start background game detection monitoring.

        Args:
            interval: Polling interval in seconds (default: 2.0).

        Returns:
            The monitoring asyncio task.
        """
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning(
                "Game monitoring already running",
                extra={"component": "detector"},
            )
            return self._monitoring_task

        self._shutdown = False
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )

        logger.info(
            f"Game monitoring started with {interval}s interval",
            extra={"component": "detector"},
        )

        return self._monitoring_task

    async def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop.

        Args:
            interval: Polling interval in seconds.
        """
        while not self._shutdown:
            try:
                await self.check_and_update()
            except Exception as e:
                logger.error(
                    f"Error in game monitoring: {e}",
                    extra={"component": "detector"},
                )
            await asyncio.sleep(interval)

    async def stop_monitoring(self) -> None:
        """Stop background game detection monitoring."""
        self._shutdown = True

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Game monitoring stopped",
            extra={"component": "detector"},
        )

    def get_active_game_ids(self) -> List[str]:
        """Get list of currently active game IDs.

        Returns:
            List of active game_ids.
        """
        return list(self._active_games)

    def is_any_game_running(self) -> bool:
        """Check if any registered game is currently running.

        Returns:
            True if at least one game is running.
        """
        return len(self._active_games) > 0
