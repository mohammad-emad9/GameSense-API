"""GameSense API - Core Services Package.

This package contains core business logic and service classes.
"""

from core.performance_monitor import PerformanceMonitor
from core.game_detector import GameDetector
from core.game_service import GameService

__all__ = [
    "PerformanceMonitor",
    "GameDetector",
    "GameService",
]

