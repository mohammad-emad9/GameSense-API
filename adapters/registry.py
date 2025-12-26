"""Adapter registry for dynamic game adapter discovery and management.

This module provides a centralized registry for game adapters,
supporting dynamic discovery and runtime registration.

Example:
    >>> from adapters.registry import AdapterRegistry
    >>> registry = AdapterRegistry()
    >>> registry.discover()
    >>> adapter = registry.get("cs2")
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from adapters.base import BaseGameAdapter
from models.game import AdapterInfo

# Configure logging
logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for game adapter discovery and management.

    Provides centralized management of game adapters with support for
    dynamic discovery, runtime registration, and lookup.

    Attributes:
        _adapters: Dictionary mapping game_id to adapter instances.
        _adapter_classes: Dictionary mapping game_id to adapter classes.

    Example:
        >>> registry = AdapterRegistry()
        >>> registry.discover()  # Auto-discover adapters
        >>> adapter = registry.get("cs2")
        >>> if adapter:
        ...     context = await adapter.get_context()
    """

    def __init__(self) -> None:
        """Initialize an empty adapter registry."""
        self._adapters: Dict[str, BaseGameAdapter] = {}
        self._adapter_classes: Dict[str, Type[BaseGameAdapter]] = {}

        logger.info(
            "AdapterRegistry initialized",
            extra={"component": "registry"},
        )

    def register(self, adapter_class: Type[BaseGameAdapter]) -> None:
        """Register an adapter class.

        Args:
            adapter_class: The adapter class to register.

        Raises:
            ValueError: If an adapter with the same game_id is already registered.
        """
        # Instantiate to get game_id
        try:
            instance = adapter_class()
            game_id = instance.game_id

            if not game_id:
                logger.warning(
                    f"Adapter {adapter_class.__name__} has no game_id, skipping",
                    extra={"component": "registry"},
                )
                return

            if game_id in self._adapters:
                logger.warning(
                    f"Adapter for '{game_id}' already registered, replacing",
                    extra={"component": "registry"},
                )

            self._adapters[game_id] = instance
            self._adapter_classes[game_id] = adapter_class

            logger.info(
                f"Registered adapter: {instance.display_name} ({game_id})",
                extra={"component": "registry", "game_id": game_id},
            )

        except Exception as e:
            logger.error(
                f"Failed to register adapter {adapter_class.__name__}: {e}",
                extra={"component": "registry"},
            )

    def unregister(self, game_id: str) -> bool:
        """Unregister an adapter by game_id.

        Args:
            game_id: The game identifier to unregister.

        Returns:
            True if adapter was unregistered, False if not found.
        """
        if game_id in self._adapters:
            del self._adapters[game_id]
            del self._adapter_classes[game_id]
            logger.info(
                f"Unregistered adapter: {game_id}",
                extra={"component": "registry"},
            )
            return True
        return False

    def get(self, game_id: str) -> Optional[BaseGameAdapter]:
        """Get an adapter instance by game_id.

        Args:
            game_id: The game identifier to look up.

        Returns:
            The adapter instance, or None if not found.
        """
        return self._adapters.get(game_id)

    def get_all(self) -> List[BaseGameAdapter]:
        """Get all registered adapter instances.

        Returns:
            List of all registered adapters.
        """
        return list(self._adapters.values())

    def list_adapters(self) -> List[AdapterInfo]:
        """Get info for all registered adapters.

        Returns:
            List of AdapterInfo for each registered adapter.
        """
        return [adapter.get_info() for adapter in self._adapters.values()]

    def list_game_ids(self) -> List[str]:
        """Get all registered game IDs.

        Returns:
            List of game_id strings.
        """
        return list(self._adapters.keys())

    def count(self) -> int:
        """Get the number of registered adapters.

        Returns:
            Number of registered adapters.
        """
        return len(self._adapters)

    def discover(self) -> int:
        """Auto-discover and register adapters from the games package.

        Scans the adapters.games package for adapter classes and
        registers them automatically.

        Returns:
            Number of adapters discovered and registered.
        """
        discovered = 0

        try:
            # Import the games package
            import adapters.games as games_package

            package_path = Path(games_package.__file__).parent

            # Iterate over modules in the games package
            for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
                if module_name.startswith("_"):
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(f"adapters.games.{module_name}")

                    # Find adapter classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # Check if it's a subclass of BaseGameAdapter
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, BaseGameAdapter)
                            and attr is not BaseGameAdapter
                            and hasattr(attr, "game_id")
                            and attr.game_id
                        ):
                            self.register(attr)
                            discovered += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to load module {module_name}: {e}",
                        extra={"component": "registry"},
                    )
                    continue

        except ImportError as e:
            logger.warning(
                f"Games package not found, no adapters discovered: {e}",
                extra={"component": "registry"},
            )

        logger.info(
            f"Discovered {discovered} adapters",
            extra={"component": "registry"},
        )

        return discovered

    def find_adapter_for_process(self, process_name: str) -> Optional[BaseGameAdapter]:
        """Find an adapter that matches a running process.

        Args:
            process_name: The process name to match (case-insensitive).

        Returns:
            The matching adapter, or None if no match found.
        """
        process_lower = process_name.lower()

        for adapter in self._adapters.values():
            for proc in adapter.process_names:
                if proc.lower() == process_lower:
                    return adapter

        return None

    def get_running_games(self) -> List[BaseGameAdapter]:
        """Get adapters for games that are currently running.

        Returns:
            List of adapters whose games are currently running.
        """
        running = []

        for adapter in self._adapters.values():
            if adapter.is_game_running():
                running.append(adapter)

        return running
