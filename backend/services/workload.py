"""Small workload controls for thermal-safe local inference."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

REALTIME_MAX_IN_FLIGHT = int(os.getenv("DEPTHLENS_REALTIME_MAX_IN_FLIGHT", "1"))
NORMAL_MAX_IN_FLIGHT = int(os.getenv("DEPTHLENS_NORMAL_MAX_IN_FLIGHT", "2"))
LOW_POWER = os.getenv("DEPTHLENS_LOW_POWER_MODE", "0").lower() in {"1", "true", "yes", "on"}

_realtime = asyncio.Semaphore(max(1, REALTIME_MAX_IN_FLIGHT))
_normal = asyncio.Semaphore(max(1, NORMAL_MAX_IN_FLIGHT))
_heavy = asyncio.Lock()
_realtime_active = 0


def low_power_status() -> dict[str, object]:
    return {
        "enabled": LOW_POWER,
        "realtime_max_in_flight": REALTIME_MAX_IN_FLIGHT,
        "normal_max_in_flight": NORMAL_MAX_IN_FLIGHT,
    }


def realtime_busy() -> bool:
    return _realtime.locked()


@asynccontextmanager
async def realtime_slot() -> AsyncIterator[None]:
    global _realtime_active
    if _realtime.locked():
        raise RuntimeError("REALTIME_BACKPRESSURE")
    await _realtime.acquire()
    _realtime_active += 1
    try:
        yield
    finally:
        _realtime_active = max(0, _realtime_active - 1)
        _realtime.release()


@asynccontextmanager
async def normal_slot() -> AsyncIterator[None]:
    await _normal.acquire()
    try:
        yield
    finally:
        _normal.release()


@asynccontextmanager
async def heavy_compute_slot() -> AsyncIterator[None]:
    await _heavy.acquire()
    try:
        yield
    finally:
        _heavy.release()
