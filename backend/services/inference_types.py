"""Shared typing helpers for depth inference services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np


class InferenceArrays(TypedDict, total=False):
    """Decoded image, normalized depth, and metadata used by response assembly."""

    img_bgr: np.ndarray
    depth: np.ndarray
    latency_ms: float
    model: str
    model_id: str
    model_display_name: str
    device_used: str
    resolution: dict[str, int]
    filename: str | None
    depth_cached: bool
    engine_requested: str
    engine_used: str
    device_requested: str
    fallback_used: bool
    fallback_reason: str | None
    onnx_path: str | None
    warnings: list[str]
    onnx: dict[str, Any] | None


class EngineMetadata(TypedDict, total=False):
    """Execution-engine metadata included in public inference responses."""

    model_id: str
    model_display_name: str
    engine_requested: str
    engine_used: str
    device_requested: str
    device_used: str
    fallback_used: bool
    fallback_reason: str | None
    onnx_path: str | None
    warnings: list[str]
    onnx: dict[str, Any] | None


@dataclass(frozen=True)
class OutputOptions:
    """Normalized output options for colorization and metrics."""

    metrics_mode: str
    outputs: tuple[str, ...]
    colormap: str


@dataclass(frozen=True)
class CacheResolution:
    """Cached depth-map resolution."""

    width: int
    height: int
