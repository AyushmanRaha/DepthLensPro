"""Redis-backed inference cache management with in-memory fallback."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pickle
import threading
import time
from typing import Any, cast

from backend.config import settings

log = logging.getLogger("depthlens")

CACHE_KEY_PREFIX = "depthlens:inference:"
CACHE_TTL_SECONDS = int(settings.CACHE_TTL_SECONDS)

_REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None
redis = importlib.import_module("redis") if _REDIS_AVAILABLE else None

_MEMORY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MEMORY_LOCK = threading.RLock()
_REDIS_LOCK = threading.Lock()
_REDIS_POOL: Any | None = None
_REDIS_CLIENT: Any | None = None
_REDIS_DISABLED_UNTIL = 0.0
_REDIS_BACKOFF_SECONDS = 10.0
_CACHE_BINARY_MAGIC = b"DLP1\0"

_METRIC_LOCK = threading.Lock()
_MEMORY_HITS = 0
_MEMORY_MISSES = 0
_REDIS_FAILURES = 0


def _redis_error_types() -> tuple[type[BaseException], ...]:
    if redis is None:
        return (OSError, TimeoutError, ConnectionError)
    return cast(tuple[type[BaseException], ...], (redis.RedisError, OSError, TimeoutError))


def _cache_error_types() -> tuple[type[BaseException], ...]:
    return _redis_error_types() + (pickle.PickleError, ValueError, TypeError)


def _redis_url() -> str:
    if settings.REDIS_URL:
        return settings.REDIS_URL

    auth = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    return f"redis://{auth}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"


def _redis_client() -> Any | None:
    """Return a shared Redis client backed by a thread-safe connection pool."""

    global _REDIS_CLIENT, _REDIS_DISABLED_UNTIL, _REDIS_POOL

    if redis is None:
        return None

    now = time.monotonic()
    if now < _REDIS_DISABLED_UNTIL:
        return None

    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    with _REDIS_LOCK:
        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT
        if time.monotonic() < _REDIS_DISABLED_UNTIL:
            return None

        _REDIS_POOL = redis.ConnectionPool.from_url(
            _redis_url(),
            socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            retry_on_timeout=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=False,
        )
        client = redis.Redis(connection_pool=_REDIS_POOL)
        try:
            client.ping()
        except _redis_error_types() as exc:
            _mark_redis_failure(exc)
            return None

        log.info("Redis cache connected at %s", _redis_url())
        _REDIS_CLIENT = client
        return _REDIS_CLIENT


def _mark_redis_failure(exc: BaseException) -> None:
    """Temporarily disable Redis and use local cache after a connection failure."""

    global _REDIS_CLIENT, _REDIS_DISABLED_UNTIL, _REDIS_FAILURES, _REDIS_POOL

    with _REDIS_LOCK:
        _REDIS_CLIENT = None
        if _REDIS_POOL is not None:
            _REDIS_POOL.disconnect()
        _REDIS_POOL = None
        _REDIS_DISABLED_UNTIL = time.monotonic() + _REDIS_BACKOFF_SECONDS

    with _METRIC_LOCK:
        _REDIS_FAILURES += 1

    log.warning("Redis cache unavailable; using in-memory fallback: %s", exc)


def _key(cache_key: str) -> str:
    return f"{CACHE_KEY_PREFIX}{cache_key}"


def _serialize(value: dict[str, Any]) -> bytes:
    """Serialize cache payloads as direct binary bytes without JSON/base64 expansion."""

    return _CACHE_BINARY_MAGIC + pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(raw: bytes) -> dict[str, Any]:
    """Deserialize current binary cache payloads with legacy base64 compatibility."""

    if raw.startswith(_CACHE_BINARY_MAGIC):
        return cast(dict[str, Any], pickle.loads(raw[len(_CACHE_BINARY_MAGIC) :]))

    # Compatibility for entries written by older DepthLens builds that wrapped
    # pickle bytes in base64, incurring unnecessary CPU and memory overhead.
    import base64

    return cast(dict[str, Any], pickle.loads(base64.b64decode(raw)))


def _memory_get(cache_key: str) -> dict[str, Any] | None:
    global _MEMORY_HITS, _MEMORY_MISSES

    now = time.time()
    with _MEMORY_LOCK:
        item = _MEMORY_CACHE.get(cache_key)
        if item is None:
            with _METRIC_LOCK:
                _MEMORY_MISSES += 1
            return None

        expires_at, value = item
        if expires_at <= now:
            _MEMORY_CACHE.pop(cache_key, None)
            with _METRIC_LOCK:
                _MEMORY_MISSES += 1
            return None

        with _METRIC_LOCK:
            _MEMORY_HITS += 1
        return value


def _memory_set(cache_key: str, value: dict[str, Any]) -> None:
    with _MEMORY_LOCK:
        _MEMORY_CACHE[cache_key] = (time.time() + CACHE_TTL_SECONDS, value)


def _memory_clear() -> int:
    with _MEMORY_LOCK:
        count = len(_MEMORY_CACHE)
        _MEMORY_CACHE.clear()
        return count


def _memory_size() -> int:
    now = time.time()
    with _MEMORY_LOCK:
        expired = [key for key, (expires_at, _) in _MEMORY_CACHE.items() if expires_at <= now]
        for key in expired:
            _MEMORY_CACHE.pop(key, None)
        return len(_MEMORY_CACHE)


def get(cache_key: str) -> dict[str, Any] | None:
    """Return a cached inference payload when present."""

    client = _redis_client()
    if client is None:
        return _memory_get(cache_key)

    try:
        raw = client.get(_key(cache_key))
        if raw is None:
            return None
        return _deserialize(raw)
    except _cache_error_types() as exc:
        _mark_redis_failure(exc)
        return _memory_get(cache_key)


def set(cache_key: str, value: dict[str, Any]) -> None:
    """Store an inference payload using the standardized cache TTL."""

    client = _redis_client()
    if client is None:
        _memory_set(cache_key, value)
        return

    try:
        client.setex(_key(cache_key), CACHE_TTL_SECONDS, _serialize(value))
    except _cache_error_types() as exc:
        _mark_redis_failure(exc)
        _memory_set(cache_key, value)


def clear() -> int:
    """Clear all cache entries and return the number removed."""

    count = _memory_clear()
    client = _redis_client()
    if client is None:
        return count

    try:
        cursor = 0
        redis_count = 0
        pattern = f"{CACHE_KEY_PREFIX}*"
        while True:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                redis_count += int(client.delete(*keys))
            if cursor == 0:
                break
        return count + redis_count
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)
        return count


def size() -> int:
    """Return the number of entries currently cached."""

    client = _redis_client()
    if client is None:
        return _memory_size()

    try:
        return sum(1 for _ in client.scan_iter(match=f"{CACHE_KEY_PREFIX}*", count=500))
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)
        return _memory_size()


def metrics() -> dict[str, Any]:
    """Return live cache hit/miss and keyspace telemetry for the dashboard."""

    memory_size = _memory_size()
    with _METRIC_LOCK:
        memory_hits = _MEMORY_HITS
        memory_misses = _MEMORY_MISSES
        redis_failures = _REDIS_FAILURES

    data: dict[str, Any] = {
        "backend": "memory",
        "redis_available": False,
        "total_hits": memory_hits,
        "cache_misses": memory_misses,
        "keyspace_size": memory_size,
        "memory_hits": memory_hits,
        "memory_misses": memory_misses,
        "memory_keyspace_size": memory_size,
        "redis_failures": redis_failures,
        "ttl_seconds": CACHE_TTL_SECONDS,
    }

    client = _redis_client()
    if client is None:
        return data

    try:
        info = client.info("stats")
        redis_hits = int(info.get("keyspace_hits", 0))
        redis_misses = int(info.get("keyspace_misses", 0))
        redis_keyspace_size = size()
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)
        return data

    data.update(
        {
            "backend": "redis",
            "redis_available": True,
            "total_hits": redis_hits + memory_hits,
            "cache_misses": redis_misses + memory_misses,
            "keyspace_size": redis_keyspace_size + memory_size,
            "redis_hits": redis_hits,
            "redis_misses": redis_misses,
            "redis_keyspace_size": redis_keyspace_size,
        }
    )
    return data
