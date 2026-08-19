"""ASGI entrypoint kept at the repository root for easy deployment."""

from app.api import app

__all__ = ["app"]
