"""Lightweight liveness routes for DepthLens Pro."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter

log = logging.getLogger("depthlens")
router = APIRouter()

SERVICE_VERSION = "3.1.0"
_SERVICE_STARTED_AT = time.time()
_FIRST_LIVE_REQUEST_LOGGED = False

log.info("LIVE_ROUTE_REGISTERED")


@router.get("/")
async def root() -> dict[str, str]:
    return {"service": "DepthLens Pro API", "version": SERVICE_VERSION}


@router.get("/live")
async def live() -> dict[str, Any]:
    global _FIRST_LIVE_REQUEST_LOGGED
    if not _FIRST_LIVE_REQUEST_LOGGED:
        log.info("FIRST_LIVE_REQUEST")
        _FIRST_LIVE_REQUEST_LOGGED = True
    now = time.time()
    return {
        "status": "ok",
        "service": "DepthLens Pro API",
        "version": SERVICE_VERSION,
        "pid": os.getpid(),
        "timestamp": now,
        "uptime_seconds": round(now - _SERVICE_STARTED_AT, 3),
    }
