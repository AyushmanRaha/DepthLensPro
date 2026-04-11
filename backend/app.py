"""Backward-compatible ASGI entrypoint for `uvicorn backend.app:app`."""

from backend.depthlens.main import app

__all__ = ["app"]
