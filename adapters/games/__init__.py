"""GameSense API - Games Adapters Package.

This package contains game-specific adapter implementations.
"""

# Import adapters to enable auto-discovery
from adapters.games.cs2 import CS2Adapter
from adapters.games.gta5 import GTA5Adapter

__all__ = [
    "CS2Adapter",
    "GTA5Adapter",
]
