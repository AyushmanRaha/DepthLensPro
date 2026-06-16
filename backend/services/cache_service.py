"""Redis-backed inference cache management with in-memory fallback."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import threading
import time
from json import JSONDecodeError
from typing import Any, cast

from backend.config import settings
from backend.services import observability

log = logging.getLogger("depthlens")

CACHE_KEY_PREFIX = "depthlens:inference:"
CACHE_TTL_SECONDS = int(settings.CACHE_TTL_SECONDS)
CACHE_MAX_ENTRIES = int(settings.CACHE_MAX_ENTRIES)

_REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None
redis = importlib.import_module("redis") if _REDIS_AVAILABLE else None

_MEMORY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MEMORY_LOCK = threading.RLock()
_REDIS_LOCK = threading.RLock()
_REDIS_POOL: Any | None = None
_REDIS_CLIENT: Any | None = None
_REDIS_DISABLED_UNTIL = 0.0
_REDIS_BACKOFF_SECONDS = 10.0
_CACHE_JSON_MAGIC = b"DLP2\0"
_LEGACY_PICKLE_MAGIC = b"DLP1\0"
_CACHE_SCHEMA_VERSION = 2

_METRIC_LOCK = threading.Lock()
_MEMORY_HITS = 0
_MEMORY_MISSES = 0
_REDIS_FAILURES = 0


class CachePayloadError(ValueError):
    """Raised when a cache payload cannot be safely encoded or decoded."""


_DESERIALIZE_ERRORS = (
    JSONDecodeError,
    UnicodeDecodeError,
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
)


def _redis_error_types() -> tuple[type[BaseException], ...]:
    if redis is None:
        return (OSError, TimeoutError, ConnectionError)
    return cast(tuple[type[BaseException], ...], (redis.RedisError, OSError, TimeoutError))


def _cache_error_types() -> tuple[type[BaseException], ...]:
    return _redis_error_types() + _DESERIALIZE_ERRORS


def _redis_url() -> str:
    if settings.REDIS_URL:
        return settings.REDIS_URL

    auth = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    return f"redis://{auth}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"


def _redis_client() -> Any | None:
    """Return a shared Redis client backed by a thread-safe connection pool."""

    global _REDIS_CLIENT, _REDIS_POOL

    if redis is None:
        return None

    with _REDIS_LOCK:
        if time.monotonic() < _REDIS_DISABLED_UNTIL:
            return None

        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT

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
        client_pool = _REDIS_POOL
        _REDIS_CLIENT = None
        _REDIS_POOL = None
        _REDIS_DISABLED_UNTIL = time.monotonic() + _REDIS_BACKOFF_SECONDS
        if client_pool is not None:
            try:
                client_pool.disconnect()
            except _redis_error_types():
                pass

    with _METRIC_LOCK:
        _REDIS_FAILURES += 1

    observability.record_cache_event("redis_error", "redis")
    observability.record_cache_event("fallback", "memory")
    log.warning("Redis cache unavailable; using in-memory fallback: %s", exc)


def _key(cache_key: str) -> str:
    return f"{CACHE_KEY_PREFIX}{cache_key}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CachePayloadError("cache payload contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CachePayloadError("cache payload contains a non-string dictionary key")
            safe[key] = _json_safe(item)
        return safe
    raise CachePayloadError(f"cache payload contains non-JSON value: {type(value).__name__}")


def _serialize(value: dict[str, Any]) -> bytes:
    """Serialize cache payloads as versioned UTF-8 JSON bytes."""

    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "value": _json_safe(value),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return _CACHE_JSON_MAGIC + encoded.encode("utf-8")


def _looks_like_legacy_pickle(raw: bytes) -> bool:
    return raw.startswith(_LEGACY_PICKLE_MAGIC) or raw.startswith(b"\x80")


def _deserialize(raw: bytes | bytearray | str) -> dict[str, Any]:
    """Deserialize versioned JSON cache bytes without executing legacy payloads."""

    if isinstance(raw, str):
        data = raw.encode("utf-8")
    else:
        data = bytes(raw)

    if _looks_like_legacy_pickle(data):
        raise CachePayloadError("legacy pickle cache payload ignored")

    if data.startswith(_CACHE_JSON_MAGIC):
        data = data[len(_CACHE_JSON_MAGIC) :]

    decoded = json.loads(data.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise CachePayloadError("cache envelope is not an object")
    if decoded.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise CachePayloadError("unsupported cache schema version")
    value = decoded["value"]
    if not isinstance(value, dict):
        raise CachePayloadError("cache value is not an object")
    return cast(dict[str, Any], _json_safe(value))


def _cleanup_expired_locked(now: float) -> None:
    expired = [key for key, (expires_at, _) in _MEMORY_CACHE.items() if expires_at <= now]
    for key in expired:
        _MEMORY_CACHE.pop(key, None)


def _enforce_memory_limit_locked() -> None:
    while len(_MEMORY_CACHE) > CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_MEMORY_CACHE), None)
        if oldest_key is None:
            break
        _MEMORY_CACHE.pop(oldest_key, None)


def _memory_get(cache_key: str) -> dict[str, Any] | None:
    global _MEMORY_HITS, _MEMORY_MISSES

    now = time.time()
    with _MEMORY_LOCK:
        _cleanup_expired_locked(now)
        item = _MEMORY_CACHE.get(cache_key)
        if item is None:
            observability.record_cache_event("miss", "memory", len(_MEMORY_CACHE))
            with _METRIC_LOCK:
                _MEMORY_MISSES += 1
            return None

        expires_at, value = item
        if expires_at <= now:
            _MEMORY_CACHE.pop(cache_key, None)
            observability.record_cache_event("miss", "memory", len(_MEMORY_CACHE))
            with _METRIC_LOCK:
                _MEMORY_MISSES += 1
            return None

        _MEMORY_CACHE.pop(cache_key, None)
        _MEMORY_CACHE[cache_key] = (expires_at, value)
        with _METRIC_LOCK:
            _MEMORY_HITS += 1
        observability.record_cache_event("hit", "memory", len(_MEMORY_CACHE))
        return value


def _memory_set(cache_key: str, value: dict[str, Any]) -> None:
    now = time.time()
    with _MEMORY_LOCK:
        _cleanup_expired_locked(now)
        _MEMORY_CACHE.pop(cache_key, None)
        _MEMORY_CACHE[cache_key] = (now + CACHE_TTL_SECONDS, value)
        _enforce_memory_limit_locked()
        observability.record_cache_event("set", "memory", len(_MEMORY_CACHE))


def _memory_clear() -> int:
    with _MEMORY_LOCK:
        count = len(_MEMORY_CACHE)
        _MEMORY_CACHE.clear()
        observability.record_cache_event("clear", "memory", 0)
        return count


def _memory_size() -> int:
    with _MEMORY_LOCK:
        _cleanup_expired_locked(time.time())
        return len(_MEMORY_CACHE)


def _delete_corrupt_redis_entry(client: Any, cache_key: str) -> None:
    try:
        client.delete(_key(cache_key))
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)


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
    except _DESERIALIZE_ERRORS as exc:
        log.warning("Ignoring corrupt cache entry for %s: %s", cache_key, exc)
        _delete_corrupt_redis_entry(client, cache_key)
        return None
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)
        return _memory_get(cache_key)


def set(cache_key: str, value: dict[str, Any]) -> None:
    """Store an inference payload using the standardized cache TTL."""

    try:
        payload = _serialize(value)
        memory_value = _deserialize(payload)
    except _DESERIALIZE_ERRORS as exc:
        observability.record_cache_event("serialization_rejected", "unknown")
        log.warning("Skipping non-JSON-safe cache payload for %s: %s", cache_key, exc)
        return

    client = _redis_client()
    if client is None:
        _memory_set(cache_key, memory_value)
        return

    try:
        client.setex(_key(cache_key), CACHE_TTL_SECONDS, payload)
        observability.record_cache_event("set", "redis")
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)
        _memory_set(cache_key, memory_value)


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
        "memory_max_entries": CACHE_MAX_ENTRIES,
    }

    client = _redis_client()
    if client is None:
        observability.record_cache_event("metrics", "memory", memory_size)
        return data

    try:
        info = client.info("stats")
        redis_hits = int(info.get("keyspace_hits", 0))
        redis_misses = int(info.get("keyspace_misses", 0))
        redis_keyspace_size = size()
    except _redis_error_types() as exc:
        _mark_redis_failure(exc)
        observability.record_cache_event("metrics", "memory", memory_size)
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
    observability.record_cache_event("metrics", "redis", data.get("keyspace_size"))
    return data
