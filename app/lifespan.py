"""Application lifespan management for GameSense API.

This module provides the FastAPI lifespan context manager that handles
startup and shutdown procedures, including initialization and cleanup
of the PerformanceMonitor and GameService.

Example:
    >>> from fastapi import FastAPI
    >>> from app.lifespan import lifespan
    >>>
    >>> app = FastAPI(lifespan=lifespan)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI

from core.performance_monitor import PerformanceMonitor
from core.game_service import GameService

# Configure structured JSON logging
logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": getattr(record, "component", "lifespan"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Apply JSON formatter if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for FastAPI application lifespan.

    Manages startup and shutdown of application services:
    - Startup: Initializes PerformanceMonitor and GameService
    - Shutdown: Cancels background tasks, cleans up resources

    Args:
        app: The FastAPI application instance.

    Yields:
        None - control returns to FastAPI during application runtime.

    Example:
        >>> app = FastAPI(lifespan=lifespan)
        >>> # Services available at app.state.performance_monitor and app.state.game_service
    """
    # === STARTUP ===
    logger.info(
        "GameSense API starting up...",
        extra={"component": "lifespan"},
    )

    # Initialize PerformanceMonitor
    performance_monitor = PerformanceMonitor()
    await performance_monitor.initialize()

    # Start background performance monitoring
    perf_task = await performance_monitor.start_background_monitoring(interval=1.0)

    # Store in app state for dependency injection
    app.state.performance_monitor = performance_monitor
    app.state.background_monitoring_task = perf_task

    logger.info(
        "Performance monitoring initialized",
        extra={"component": "lifespan"},
    )

    # Initialize GameService
    game_service = GameService()
    await game_service.initialize()

    # Start background game detection
    game_task = await game_service.start_monitoring(interval=2.0)

    # Store in app state
    app.state.game_service = game_service
    app.state.game_monitoring_task = game_task

    logger.info(
        "Game service initialized",
        extra={"component": "lifespan"},
    )

    logger.info(
        "GameSense API startup complete - All services active",
        extra={"component": "lifespan"},
    )

    try:
        yield
    finally:
        # === SHUTDOWN ===
        logger.info(
            "GameSense API shutting down...",
            extra={"component": "lifespan"},
        )

        # Cleanup GameService
        if hasattr(app.state, "game_service"):
            await app.state.game_service.cleanup()
            logger.info(
                "Game service cleanup complete",
                extra={"component": "lifespan"},
            )

        # Cleanup PerformanceMonitor
        if hasattr(app.state, "performance_monitor"):
            await app.state.performance_monitor.cleanup()
            logger.info(
                "Performance monitor cleanup complete",
                extra={"component": "lifespan"},
            )

        logger.info(
            "GameSense API shutdown complete",
            extra={"component": "lifespan"},
        )

