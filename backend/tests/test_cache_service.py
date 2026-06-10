"""Cache-service security and correctness tests."""

from __future__ import annotations

import pickle
import threading
import time
from typing import Any

import pytest

from backend.services import cache_service
from backend.services.inference import _fhash, _raw_hash


@pytest.fixture(autouse=True)
def reset_cache_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_service, "redis", None)
    monkeypatch.setattr(cache_service, "_REDIS_CLIENT", None)
    monkeypatch.setattr(cache_service, "_REDIS_POOL", None)
    monkeypatch.setattr(cache_service, "_REDIS_DISABLED_UNTIL", 0.0)
    cache_service._memory_clear()
    yield
    cache_service._memory_clear()
    monkeypatch.setattr(cache_service, "_REDIS_CLIENT", None)
    monkeypatch.setattr(cache_service, "_REDIS_POOL", None)
    monkeypatch.setattr(cache_service, "_REDIS_DISABLED_UNTIL", 0.0)


def test_json_serialization_round_trip() -> None:
    value = {
        "depth_map": "iVBORw0KGgo=",
        "metrics": {"mean": 0.5, "valid": True},
        "outputs": ["color", "gray"],
        "count": 2,
        "optional": None,
    }

    encoded = cache_service._serialize(value)

    assert isinstance(encoded, bytes)
    assert encoded.startswith(cache_service._CACHE_JSON_MAGIC)
    assert cache_service._deserialize(encoded) == value


def test_redis_deserialization_never_calls_pickle_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_loads(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pickle.loads must not be used for Redis cache data")

    monkeypatch.setattr(pickle, "loads", fail_loads)
    value = {"depth_map": "png", "metrics": {"mean": 1.0}}

    assert cache_service._deserialize(cache_service._serialize(value)) == value


def test_legacy_pickle_cache_value_is_miss_and_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def get(self, key: str) -> bytes:
            return cache_service._LEGACY_PICKLE_MAGIC + b"legacy"

        def delete(self, key: str) -> int:
            self.deleted.append(key)
            return 1

    fake = FakeRedis()
    monkeypatch.setattr(cache_service, "_redis_client", lambda: fake)

    assert cache_service.get("legacy") is None
    assert fake.deleted == [cache_service._key("legacy")]


def test_corrupt_binary_cache_value_is_miss_and_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.deleted = False

        def get(self, key: str) -> bytes:
            return b"\xff\x00not-json"

        def delete(self, key: str) -> int:
            self.deleted = True
            return 1

    fake = FakeRedis()
    monkeypatch.setattr(cache_service, "_redis_client", lambda: fake)

    assert cache_service.get("corrupt") is None
    assert fake.deleted is True


def test_memory_cache_enforces_max_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_service, "CACHE_MAX_ENTRIES", 2)

    cache_service.set("one", {"value": 1})
    cache_service.set("two", {"value": 2})
    assert cache_service.get("one") == {"value": 1}
    cache_service.set("three", {"value": 3})

    assert cache_service.size() == 2
    assert cache_service.get("two") is None
    assert cache_service.get("one") == {"value": 1}
    assert cache_service.get("three") == {"value": 3}


def test_expired_memory_entries_are_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_service, "CACHE_TTL_SECONDS", 1)
    cache_service.set("expired", {"value": True})
    with cache_service._MEMORY_LOCK:
        expires_at, value = cache_service._MEMORY_CACHE["expired"]
        cache_service._MEMORY_CACHE["expired"] = (time.time() - 1, value)

    assert cache_service.get("expired") is None
    assert cache_service.size() == 0


def test_sha256_cache_key_changes_when_raw_bytes_change() -> None:
    first = b"image-a"
    second = b"image-b"

    assert _raw_hash(first) != _raw_hash(second)
    assert len(_raw_hash(first)) == 64
    assert _fhash(first, "MiDaS_small", "inferno", "cpu") != _fhash(
        second, "MiDaS_small", "inferno", "cpu"
    )


def test_redis_failure_backoff_is_safe_under_concurrent_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRedisError(Exception):
        pass

    class FailingClient:
        def get(self, key: str) -> bytes:
            raise FakeRedisError("boom")

    cache_service.set("fallback", {"ok": True})
    monkeypatch.setattr(cache_service, "redis", object())
    monkeypatch.setattr(cache_service, "_redis_error_types", lambda: (FakeRedisError,))
    monkeypatch.setattr(cache_service, "_REDIS_CLIENT", FailingClient())
    monkeypatch.setattr(cache_service, "_REDIS_DISABLED_UNTIL", 0.0)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            cache_service.get("fallback")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert cache_service._REDIS_CLIENT is None
    assert cache_service._REDIS_DISABLED_UNTIL > time.monotonic()
