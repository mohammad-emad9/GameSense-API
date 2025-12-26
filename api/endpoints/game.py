"""Game telemetry API endpoints for GameSense API.

This module provides REST API endpoints for game detection and
telemetry collection.

Endpoints:
    GET /api/v1/game/ - Get active game context
    GET /api/v1/game/adapters - List registered adapters
    GET /api/v1/game/{game_id} - Get specific game context

Example:
    >>> # Start server: uvicorn main:app
    >>> # GET http://localhost:8000/api/v1/game/
    >>> # Response: {"detected": true, "game": {...}}
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from core.game_service import GameService
from models.game import ActiveGameResponse, AdapterInfo, GameContext

router = APIRouter(
    prefix="/api/v1/game",
    tags=["Game"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable - game service not initialized"},
    },
)


async def get_game_service(request: Request) -> GameService:
    """Dependency to get the GameService instance from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The GameService instance.

    Raises:
        HTTPException: If the game service is not initialized (503).
    """
    service = getattr(request.app.state, "game_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Game service not initialized. Server may be starting up.",
        )
    return service


# Type alias for dependency injection
GameServiceDep = Annotated[GameService, Depends(get_game_service)]


@router.get(
    "/",
    response_model=ActiveGameResponse,
    summary="Get Active Game",
    description="Detect and return the currently active game with telemetry data.",
    response_description="Active game detection result with context",
)
async def get_active_game(
    service: GameServiceDep,
) -> ActiveGameResponse:
    """Get the currently active game context.

    Detects any running game from registered adapters and returns
    telemetry data collected from available sources.

    Args:
        service: Injected GameService instance.

    Returns:
        ActiveGameResponse with detection status and game context.

    Raises:
        HTTPException: If detection fails (500).

    Example Response:
        ```json
        {
            "detected": true,
            "game": {
                "game_id": "cs2",
                "active": true,
                "state": {"round_phase": "live"},
                "stats": {"kills": 5, "deaths": 2},
                "metadata": {"map": "de_dust2"}
            },
            "available_adapters": 2
        }
        ```
    """
    try:
        return await service.get_active_game()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect active game: {str(e)}",
        ) from e


@router.get(
    "/adapters",
    response_model=List[AdapterInfo],
    summary="List Game Adapters",
    description="Get information about all registered game adapters.",
    response_description="List of adapter info",
)
async def list_adapters(
    service: GameServiceDep,
) -> List[AdapterInfo]:
    """List all registered game adapters.

    Returns information about each adapter including supported
    telemetry strategies and process names.

    Args:
        service: Injected GameService instance.

    Returns:
        List of AdapterInfo for each registered adapter.

    Example Response:
        ```json
        [
            {
                "id": "cs2",
                "name": "Counter-Strike 2",
                "version": "1.0.0",
                "supported_strategies": ["log", "window"],
                "process_names": ["cs2.exe"]
            }
        ]
        ```
    """
    return service.list_adapters()


@router.get(
    "/running",
    response_model=List[str],
    summary="Get Running Games",
    description="Get list of currently running game IDs.",
    response_description="List of running game IDs",
)
async def get_running_games(
    service: GameServiceDep,
) -> List[str]:
    """Get list of currently running games.

    Args:
        service: Injected GameService instance.

    Returns:
        List of game_id strings for running games.

    Example Response:
        ```json
        ["cs2"]
        ```
    """
    return service.get_running_game_ids()


@router.get(
    "/{game_id}",
    response_model=GameContext,
    summary="Get Game Context",
    description="Get telemetry context for a specific game.",
    response_description="Game context with telemetry data",
    responses={
        404: {"description": "Game adapter not found"},
    },
)
async def get_game_context(
    service: GameServiceDep,
    game_id: str,
) -> GameContext:
    """Get context for a specific game.

    Args:
        service: Injected GameService instance.
        game_id: The game identifier (e.g., "cs2", "gta5").

    Returns:
        GameContext with telemetry data.

    Raises:
        HTTPException: If game not found (404) or query fails (500).

    Example Response:
        ```json
        {
            "game_id": "cs2",
            "active": true,
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "state": {"round_phase": "live", "bomb_planted": false},
            "stats": {"kills": 12, "deaths": 5},
            "metadata": {"map": "de_inferno"},
            "sources_used": ["log", "window"]
        }
        ```
    """
    try:
        context = await service.get_game_context(game_id)

        if context is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game adapter not found: {game_id}",
            )

        return context

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get game context: {str(e)}",
        ) from e


@router.get(
    "/{game_id}/info",
    response_model=AdapterInfo,
    summary="Get Adapter Info",
    description="Get information about a specific game adapter.",
    response_description="Adapter information",
    responses={
        404: {"description": "Adapter not found"},
    },
)
async def get_adapter_info(
    service: GameServiceDep,
    game_id: str,
) -> AdapterInfo:
    """Get information about a specific adapter.

    Args:
        service: Injected GameService instance.
        game_id: The game identifier.

    Returns:
        AdapterInfo with adapter details.

    Raises:
        HTTPException: If adapter not found (404).
    """
    info = service.get_adapter_info(game_id)

    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Adapter not found: {game_id}",
        )

    return info


@router.get(
    "/{game_id}/running",
    response_model=dict,
    summary="Check Game Running",
    description="Check if a specific game is currently running.",
    response_description="Running status",
)
async def check_game_running(
    service: GameServiceDep,
    game_id: str,
) -> dict:
    """Check if a specific game is running.

    Args:
        service: Injected GameService instance.
        game_id: The game identifier.

    Returns:
        Dict with 'running' boolean status.

    Example Response:
        ```json
        {
            "game_id": "cs2",
            "running": true
        }
        ```
    """
    return {
        "game_id": game_id,
        "running": service.is_game_running(game_id),
    }
