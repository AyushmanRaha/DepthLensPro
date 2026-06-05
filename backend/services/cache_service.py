"""In-memory inference cache management."""

from __future__ import annotations

from typing import Any

CACHE: dict[str, dict[str, Any]] = {}


def get(cache_key: str) -> dict[str, Any] | None:
    """Return a cached inference payload when present."""
    return CACHE.get(cache_key)


def set(cache_key: str, value: dict[str, Any]) -> None:
    """Store an inference payload in the in-memory cache."""
    CACHE[cache_key] = value


def clear() -> int:
    """Clear all cache entries and return the number removed."""
    count = len(CACHE)
    CACHE.clear()
    return count


def size() -> int:
    """Return the number of entries currently cached."""
    return len(CACHE)
