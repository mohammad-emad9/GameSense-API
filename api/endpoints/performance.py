"""Performance monitoring API endpoints for GameSense API.

This module provides REST API endpoints for retrieving real-time system
performance metrics and bottleneck analysis.

Endpoints:
    GET /api/v1/performance/ - Get current system performance stats
    GET /api/v1/performance/bottleneck - Detect performance bottlenecks

Example:
    >>> # Start server: uvicorn main:app
    >>> # GET http://localhost:8000/api/v1/performance/
    >>> # Response: {"cpu": {...}, "gpu": {...}, ...}
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from core.performance_monitor import PerformanceMonitor
from models.performance import BottleneckAnalysis, SystemStats

router = APIRouter(
    prefix="/api/v1/performance",
    tags=["Performance"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable - monitor not initialized"},
    },
)


async def get_performance_monitor(request: Request) -> PerformanceMonitor:
    """Dependency to get the PerformanceMonitor instance from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The PerformanceMonitor instance.

    Raises:
        HTTPException: If the monitor is not initialized (503).
    """
    monitor = getattr(request.app.state, "performance_monitor", None)
    if monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Performance monitor not initialized. Server may be starting up.",
        )
    return monitor


# Type alias for dependency injection
PerformanceMonitorDep = Annotated[PerformanceMonitor, Depends(get_performance_monitor)]


@router.get(
    "/",
    response_model=SystemStats,
    summary="Get Current Performance",
    description="Retrieve real-time system performance statistics including CPU, GPU, memory, disk I/O, and network metrics.",
    response_description="Current system performance statistics",
)
async def get_current_performance(
    monitor: PerformanceMonitorDep,
    use_cache: bool = False,
) -> SystemStats:
    """Get current system performance statistics.

    Collects and returns comprehensive performance metrics from all
    monitored subsystems (CPU, GPU, memory, disk, network).

    Args:
        monitor: Injected PerformanceMonitor instance.
        use_cache: If True, return cached stats without fresh measurement.

    Returns:
        SystemStats containing all current performance metrics.

    Raises:
        HTTPException: If stats collection fails (500).

    Example Response:
        ```json
        {
            "cpu": {"usage_percent": 45.2, "core_count": 8},
            "gpu": {"utilization": 70.0, "temperature": 65.0},
            "memory": {"used": 8589934592, "total": 17179869184, "percent": 50.0},
            "disk": {"read_speed_mbps": 150.5, "write_speed_mbps": 120.3},
            "network": {"latency_ms": 25.0, "jitter_ms": 3.5},
            "timestamp": "2024-01-15T10:30:00Z"
        }
        ```
    """
    try:
        if use_cache:
            cached = await monitor.get_cached_stats()
            if cached is not None:
                return cached
        return await monitor.get_system_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect performance stats: {str(e)}",
        ) from e


@router.get(
    "/bottleneck",
    response_model=BottleneckAnalysis,
    summary="Detect Performance Bottleneck",
    description="Analyze system performance to identify potential bottlenecks limiting performance.",
    response_description="Bottleneck analysis with type, severity, and recommendations",
)
async def detect_bottleneck(
    monitor: PerformanceMonitorDep,
) -> BottleneckAnalysis:
    """Detect current performance bottleneck.

    Analyzes CPU, GPU, memory, and network metrics to identify what
    component is most likely limiting system performance.

    Args:
        monitor: Injected PerformanceMonitor instance.

    Returns:
        BottleneckAnalysis with bottleneck type, severity, and recommendations.

    Raises:
        HTTPException: If analysis fails (500).

    Example Response:
        ```json
        {
            "bottleneck_type": "gpu_limited",
            "severity": 0.8,
            "description": "GPU is at 95% utilization while CPU is at 45%",
            "recommendations": [
                "Lower resolution or graphics quality",
                "Enable DLSS/FSR if supported"
            ]
        }
        ```
    """
    try:
        return await monitor.detect_bottleneck()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze bottleneck: {str(e)}",
        ) from e


@router.get(
    "/throttling",
    response_model=dict,
    summary="Check Thermal Throttling",
    description="Check if the system is experiencing thermal throttling.",
    response_description="Thermal throttling status",
)
async def check_thermal_throttling(
    monitor: PerformanceMonitorDep,
) -> dict:
    """Check if thermal throttling is occurring.

    Uses heuristic detection based on high temperatures combined with
    sudden utilization drops.

    Args:
        monitor: Injected PerformanceMonitor instance.

    Returns:
        Dict with 'is_throttling' boolean flag.

    Raises:
        HTTPException: If check fails (500).

    Example Response:
        ```json
        {
            "is_throttling": false
        }
        ```
    """
    try:
        is_throttling = await monitor.is_thermal_throttling()
        return {"is_throttling": is_throttling}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check thermal throttling: {str(e)}",
        ) from e
