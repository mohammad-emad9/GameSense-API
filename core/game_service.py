"""Game service for GameSense API.

This module provides the main game service that orchestrates
adapters, detection, and telemetry collection.

Example:
    >>> from core.game_service import GameService
    >>> service = GameService()
    >>> await service.initialize()
    >>> context = await service.get_active_game_context()
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from adapters.base import BaseGameAdapter
from adapters.registry import AdapterRegistry
from core.game_detector import GameDetector
from models.game import ActiveGameResponse, AdapterInfo, GameContext

# Configure logging
logger = logging.getLogger(__name__)


class GameService:
    """Main service for game telemetry collection.

    Orchestrates adapter discovery, game detection, and context
    collection. Provides a unified interface for game telemetry.

    Attributes:
        _registry: Adapter registry for managing game adapters.
        _detector: Game detector for identifying running games.
        _cache: Cached game contexts.
        _cache_lock: Asyncio lock for thread-safe cache access.

    Example:
        >>> service = GameService()
        >>> await service.initialize()
        >>> response = await service.get_active_game()
        >>> if response.detected:
        ...     print(f"Playing: {response.game.metadata.get('display_name')}")
    """

    def __init__(self) -> None:
        """Initialize the game service."""
        self._registry = AdapterRegistry()
        self._detector: Optional[GameDetector] = None
        self._cache: Dict[str, GameContext] = {}
        self._cache_lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._initialized = False

        logger.info(
            "GameService instance created",
            extra={"component": "game_service"},
        )

    async def initialize(self) -> None:
        """Initialize the game service.

        Discovers adapters and sets up the game detector.
        """
        if self._initialized:
            logger.warning(
                "GameService already initialized",
                extra={"component": "game_service"},
            )
            return

        # Discover adapters
        discovered = self._registry.discover()
        logger.info(
            f"Discovered {discovered} game adapters",
            extra={"component": "game_service"},
        )

        # Initialize detector
        self._detector = GameDetector(self._registry)

        # Set up event handlers
        self._detector.on_game_start(self._on_game_start)
        self._detector.on_game_stop(self._on_game_stop)

        self._initialized = True

        logger.info(
            "GameService initialized successfully",
            extra={"component": "game_service"},
        )

    def _on_game_start(self, game_id: str) -> None:
        """Handle game start event.

        Args:
            game_id: The game that started.
        """
        logger.info(
            f"Game service detected game start: {game_id}",
            extra={"component": "game_service", "game_id": game_id},
        )

    def _on_game_stop(self, game_id: str) -> None:
        """Handle game stop event.

        Args:
            game_id: The game that stopped.
        """
        # Clear cache for stopped game
        if game_id in self._cache:
            del self._cache[game_id]

        logger.info(
            f"Game service detected game stop: {game_id}",
            extra={"component": "game_service", "game_id": game_id},
        )

    async def start_monitoring(self, interval: float = 2.0) -> asyncio.Task:
        """Start background game monitoring.

        Args:
            interval: Detection polling interval in seconds.

        Returns:
            The monitoring task.
        """
        if not self._detector:
            raise RuntimeError("GameService not initialized. Call initialize() first.")

        self._monitoring_task = await self._detector.start_monitoring(interval)
        return self._monitoring_task

    async def stop_monitoring(self) -> None:
        """Stop background game monitoring."""
        if self._detector:
            await self._detector.stop_monitoring()

    async def cleanup(self) -> None:
        """Cleanup game service resources."""
        await self.stop_monitoring()

        # Clear cache
        async with self._cache_lock:
            self._cache.clear()

        logger.info(
            "GameService cleanup complete",
            extra={"component": "game_service"},
        )

    async def get_active_game(self) -> ActiveGameResponse:
        """Get the currently active game context.

        Returns:
            ActiveGameResponse with detection status and game context.
        """
        if not self._detector:
            return ActiveGameResponse(
                detected=False,
                available_adapters=0,
            )

        adapter = self._detector.get_active_adapter()

        if adapter is None:
            return ActiveGameResponse(
                detected=False,
                available_adapters=self._registry.count(),
            )

        # Get context from adapter
        context = await adapter.get_context()

        # Cache the context
        async with self._cache_lock:
            self._cache[adapter.game_id] = context

        return ActiveGameResponse(
            detected=True,
            game=context,
            available_adapters=self._registry.count(),
        )

    async def get_game_context(self, game_id: str) -> Optional[GameContext]:
        """Get context for a specific game.

        Args:
            game_id: The game identifier to get context for.

        Returns:
            GameContext if game is found and running, None otherwise.
        """
        adapter = self._registry.get(game_id)

        if adapter is None:
            logger.warning(
                f"No adapter found for game: {game_id}",
                extra={"component": "game_service", "game_id": game_id},
            )
            return None

        if not adapter.is_game_running():
            # Return cached context if available
            async with self._cache_lock:
                return self._cache.get(game_id)

        # Get fresh context
        context = await adapter.get_context()

        # Update cache
        async with self._cache_lock:
            self._cache[game_id] = context

        return context

    async def get_cached_context(self, game_id: str) -> Optional[GameContext]:
        """Get cached context without querying the adapter.

        Args:
            game_id: The game identifier.

        Returns:
            Cached GameContext, or None if not cached.
        """
        async with self._cache_lock:
            return self._cache.get(game_id)

    def list_adapters(self) -> List[AdapterInfo]:
        """Get information about all registered adapters.

        Returns:
            List of AdapterInfo for each registered adapter.
        """
        return self._registry.list_adapters()

    def get_adapter_info(self, game_id: str) -> Optional[AdapterInfo]:
        """Get information about a specific adapter.

        Args:
            game_id: The game identifier.

        Returns:
            AdapterInfo if found, None otherwise.
        """
        adapter = self._registry.get(game_id)
        return adapter.get_info() if adapter else None

    def get_running_game_ids(self) -> List[str]:
        """Get list of currently running game IDs.

        Returns:
            List of running game_ids.
        """
        if self._detector:
            return self._detector.get_active_game_ids()
        return self._registry.list_game_ids()

    def is_game_running(self, game_id: str) -> bool:
        """Check if a specific game is running.

        Args:
            game_id: The game identifier to check.

        Returns:
            True if game is running, False otherwise.
        """
        adapter = self._registry.get(game_id)
        return adapter.is_game_running() if adapter else False
