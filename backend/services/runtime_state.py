"""Tiny process runtime state for lightweight liveness checks.

This module must remain import-light: do not import inference runtimes or scan assets here.
"""

from __future__ import annotations

import threading

_BENCHMARK_COUNT = 0
_LOCK = threading.Lock()


def benchmark_busy() -> bool:
    with _LOCK:
        return _BENCHMARK_COUNT > 0


def set_benchmark_busy(active: bool) -> None:
    global _BENCHMARK_COUNT
    with _LOCK:
        if active:
            _BENCHMARK_COUNT += 1
        else:
            _BENCHMARK_COUNT = max(0, _BENCHMARK_COUNT - 1)
