"""GameSense API - Main Application Entry Point.

A high-performance desktop gaming assistant backend API for Windows,
providing real-time performance telemetry, game detection, bottleneck
detection, and thermal monitoring.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Example:
    $ curl http://localhost:8000/api/v1/performance/
    $ curl http://localhost:8000/api/v1/game/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.endpoints.performance import router as performance_router
from api.endpoints.game import router as game_router
from app.lifespan import lifespan

# Application metadata
APP_TITLE = "GameSense API"
APP_DESCRIPTION = """
## High-Performance Gaming Assistant Backend

GameSense API provides real-time system performance monitoring and game
telemetry collection optimized for Windows gaming environments.

### Features

- **Real-time Performance Telemetry**: CPU, GPU, Memory, Disk I/O, Network stats
- **GPU Support**: NVIDIA (via NVML), AMD/Intel (via WMI fallback)
- **Bottleneck Detection**: Intelligent analysis of limiting factors
- **Thermal Monitoring**: Throttling detection with recommendations
- **Game Detection**: Automatic game detection with adapter plugins
- **Game Telemetry**: CS2, GTA V, and extensible for more games
- **Low Overhead**: Async-first design with thread-safe caching

### Getting Started

1. Start the server: `uvicorn main:app --reload`
2. Open API docs: http://localhost:8000/docs
3. Query performance: `GET /api/v1/performance/`
4. Query active game: `GET /api/v1/game/`
"""
APP_VERSION = "2.0.0"

# Create FastAPI application with lifespan management
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(performance_router)
app.include_router(game_router)


@app.get(
    "/",
    response_class=JSONResponse,
    tags=["Root"],
    summary="API Root",
    description="Returns API information and health status.",
)
async def root() -> dict:
    """API root endpoint.

    Returns:
        Dict with API name, version, and documentation links.
    """
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "endpoints": {
            "performance": "/api/v1/performance/",
            "bottleneck": "/api/v1/performance/bottleneck",
            "throttling": "/api/v1/performance/throttling",
            "game": "/api/v1/game/",
            "adapters": "/api/v1/game/adapters",
        },
    }


@app.get(
    "/health",
    response_class=JSONResponse,
    tags=["Health"],
    summary="Health Check",
    description="Returns service health status.",
)
async def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring.

    Returns:
        Dict with health status.
    """
    return {"status": "healthy", "service": APP_TITLE, "version": APP_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )

