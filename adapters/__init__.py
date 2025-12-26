"""GameSense API - Adapters Package.

This package contains the game adapter plugin system for universal
game telemetry collection.
"""

from adapters.base import BaseGameAdapter, BaseTelemetryStrategy
from adapters.registry import AdapterRegistry

__all__ = [
    "BaseGameAdapter",
    "BaseTelemetryStrategy",
    "AdapterRegistry",
]
