"""Depth inference cache and stable cache-key helpers."""

from __future__ import annotations

import hashlib
import threading
import time

import numpy as np

from backend.config import settings
from backend.services.image_io import MAX_DIM
from backend.services.metrics import normalize_metrics_mode, parse_outputs

_DEPTH_CACHE: dict[str, tuple[float, np.ndarray, dict[str, int]]] = {}
_DEPTH_CACHE_LOCK = threading.RLock()
_DEPTH_CACHE_MAX_ENTRIES = 12


def _raw_hash(raw: bytes) -> str:
    """Return a stable content hash for raw image bytes."""
    return hashlib.sha256(raw).hexdigest()


def _depth_cache_key(
    raw: bytes, model: str, dev: str, max_dim: int | None, engine: str = "auto"
) -> str:
    """Cache normalized depth independently from color/output/metric options."""
    limit = max(256, int(max_dim or MAX_DIM))
    return hashlib.sha1(
        f"depth:{model}:{dev}:{engine}:{limit}:{_raw_hash(raw)}".encode()
    ).hexdigest()


def _fhash(
    raw: bytes,
    model: str,
    cmap: str,
    dev: str,
    metrics: str = "full",
    outputs: str = "color,gray",
    max_dim: int | None = None,
    engine: str = "auto",
) -> str:
    """Build a stable cache key for request image bytes and response options."""
    limit = max(256, int(max_dim or MAX_DIM))
    output_key = ",".join(parse_outputs(outputs))
    metric_key = normalize_metrics_mode(metrics)
    return hashlib.sha1(
        f"{model}:{cmap}:{dev}:{engine}:{metric_key}:{output_key}:{limit}:{_raw_hash(raw)}".encode()
    ).hexdigest()


def _get_cached_depth(cache_key: str) -> tuple[np.ndarray, dict[str, int]] | None:
    now = time.time()
    with _DEPTH_CACHE_LOCK:
        item = _DEPTH_CACHE.get(cache_key)
        if item is None:
            return None
        expires_at, depth, resolution = item
        if expires_at <= now:
            _DEPTH_CACHE.pop(cache_key, None)
            return None
        return depth.copy(), dict(resolution)


def _set_cached_depth(cache_key: str, depth: np.ndarray, resolution: dict[str, int]) -> None:
    with _DEPTH_CACHE_LOCK:
        if len(_DEPTH_CACHE) >= _DEPTH_CACHE_MAX_ENTRIES:
            oldest = min(_DEPTH_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _DEPTH_CACHE.pop(oldest, None)
        _DEPTH_CACHE[cache_key] = (
            time.time() + int(settings.CACHE_TTL_SECONDS),
            depth.copy(),
            dict(resolution),
        )
