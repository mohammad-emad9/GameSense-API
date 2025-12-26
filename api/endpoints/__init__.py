"""GameSense API - Endpoints Package.

This package contains all API endpoint routers.
"""

from api.endpoints.performance import router as performance_router
from api.endpoints.game import router as game_router

__all__ = [
    "performance_router",
    "game_router",
]

