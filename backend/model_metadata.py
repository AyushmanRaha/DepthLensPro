"""Backward-compatible lightweight metadata aliases.

The canonical model data lives in :mod:`backend.model_registry`.  This module is
kept only for older imports and intentionally imports no ML runtime packages.
"""

from __future__ import annotations

from backend.model_registry import supported_models_legacy_payload

SUPPORTED_MODELS: dict[str, dict[str, str]] = supported_models_legacy_payload()

COLORMAP_NAMES: tuple[str, ...] = (
    "inferno",
    "plasma",
    "viridis",
    "magma",
    "jet",
    "hot",
    "bone",
    "turbo",
)
