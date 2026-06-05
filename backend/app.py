"""Backward-compatible ASGI entrypoint for existing DepthLens Pro launch flows."""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from backend.main import app  # noqa: E402

__all__ = ["app"]
