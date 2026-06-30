"""Tiny process runtime state for lightweight liveness checks.

This module must remain import-light: stdlib only; do not import inference runtimes
or scan assets here.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

_KINDS = {
    "inference": "active_inference",
    "compare": "active_compare",
    "benchmark": "active_benchmark",
    "reconstruction": "active_reconstruction",
    "detection": "active_detection",
}
_LOCK = threading.RLock()
_ACTIVE = {value: 0 for value in _KINDS.values()}
_LAST_OPERATION_NAME: str | None = None
_LAST_OPERATION_STARTED_AT: float | None = None
_LAST_OPERATION_FINISHED_AT: float | None = None


def _key(kind: str) -> str:
    normalized = (kind or "").strip().lower().replace("-", "_")
    return _KINDS.get(normalized, f"active_{normalized or 'operation'}")


def start_operation(kind: str, label: str | None = None) -> None:
    global _LAST_OPERATION_NAME, _LAST_OPERATION_STARTED_AT
    now = time.time()
    key = _key(kind)
    name = label or kind
    with _LOCK:
        _ACTIVE[key] = int(_ACTIVE.get(key, 0)) + 1
        _LAST_OPERATION_NAME = name
        _LAST_OPERATION_STARTED_AT = now


def end_operation(kind: str) -> None:
    global _LAST_OPERATION_FINISHED_AT
    key = _key(kind)
    with _LOCK:
        _ACTIVE[key] = max(0, int(_ACTIVE.get(key, 0)) - 1)
        _LAST_OPERATION_FINISHED_AT = time.time()


@contextmanager
def track_operation(kind: str, label: str | None = None) -> Iterator[None]:
    start_operation(kind, label)
    try:
        yield
    finally:
        end_operation(kind)


def snapshot() -> dict[str, Any]:
    with _LOCK:
        active = {key: int(_ACTIVE.get(key, 0)) for key in sorted(_ACTIVE)}
        total = sum(active.values())
        return {
            "busy": total > 0,
            "active_operations": active,
            "total_active_operations": total,
            "last_operation": _LAST_OPERATION_NAME,
            "last_operation_started_at": _LAST_OPERATION_STARTED_AT,
            "last_operation_finished_at": _LAST_OPERATION_FINISHED_AT,
        }


def benchmark_busy() -> bool:
    active = snapshot()["active_operations"]
    return bool(active.get("active_benchmark", 0) > 0)


def set_benchmark_busy(active: bool) -> None:
    if active:
        start_operation("benchmark", "benchmark")
    else:
        end_operation("benchmark")
