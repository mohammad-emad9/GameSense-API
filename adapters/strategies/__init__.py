"""GameSense API - Telemetry Strategies Package.

This package contains telemetry data collection strategies.
"""

from adapters.strategies.base import BaseTelemetryStrategy
from adapters.strategies.log_reader import LogReaderStrategy
from adapters.strategies.window_observer import WindowObserverStrategy

__all__ = [
    "BaseTelemetryStrategy",
    "LogReaderStrategy",
    "WindowObserverStrategy",
]
