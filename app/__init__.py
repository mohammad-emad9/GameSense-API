"""GameSense API - Application Package.

This package contains application lifecycle and configuration.
"""

from app.lifespan import lifespan

__all__ = ["lifespan"]
