"""GameSense API - Models Package.

This package contains Pydantic models for performance and game telemetry data.
"""

from models.performance import (
    CPUStats,
    GPUStats,
    MemoryStats,
    DiskStats,
    NetworkStats,
    SystemStats,
    BottleneckAnalysis,
    BottleneckType,
)
from models.game import (
    GameContext,
    AdapterInfo,
    TelemetrySource,
    ActiveGameResponse,
)

__all__ = [
    # Performance models
    "CPUStats",
    "GPUStats",
    "MemoryStats",
    "DiskStats",
    "NetworkStats",
    "SystemStats",
    "BottleneckAnalysis",
    "BottleneckType",
    # Game models
    "GameContext",
    "AdapterInfo",
    "TelemetrySource",
    "ActiveGameResponse",
]

