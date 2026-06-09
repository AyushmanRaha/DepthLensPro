"""FastAPI application factory and runtime lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.config import settings
from backend.services import cache_service
from backend.services.inference import SUPPORTED_MODELS, _load_model, clear_models
from backend.utils.hardware import _available_devices, _default_device_key, _resolve


class JsonLogFormatter(logging.Formatter):
    """Format Python log records as structured JSON for production collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack_trace": "".join(traceback.format_exception(*record.exc_info)),
            }
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_level: str) -> None:
    """Route application and server logs through a single JSON formatter."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level.upper())

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True


configure_logging(settings.LOG_LEVEL)
log = logging.getLogger("depthlens")


async def _warm_default_model() -> None:
    """Optionally warm one model after the ASGI app is already serving /live."""
    model = settings.DEPTHLENS_WARMUP_MODEL
    requested_device = settings.DEPTHLENS_WARMUP_DEVICE
    if model not in SUPPORTED_MODELS:
        log.warning("⚠️  Warmup skipped: unknown model %s", model)
        return
    try:
        device = str(_resolve(requested_device))
    except Exception as exc:
        log.warning(
            "⚠️  Warmup device %s unavailable; falling back to CPU: %s", requested_device, exc
        )
        device = "cpu"
    try:
        log.info("🔥 Background warmup starting: %s on %s", model, device)
        await asyncio.to_thread(_load_model, model, device)
        log.info("✅ Background warmup complete: %s on %s", model, device)
    except Exception as exc:
        log.warning("⚠️  Background warmup failed for %s on %s: %s", model, device, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start quickly; optional model warmup runs after liveness is available."""
    try:
        devs = _available_devices()
        best = _default_device_key()
        log.info("🚀 DepthLens Pro v3.1 — devices: %s  best: %s", list(devs), best)
    except Exception as exc:
        log.warning("⚠️  Device discovery degraded during startup: %s", exc)
    warmup_task: asyncio.Task[None] | None = None
    if settings.DEPTHLENS_PRELOAD_MODEL:
        warmup_task = asyncio.create_task(_warm_default_model())
    else:
        log.info("Model preload disabled; first inference will lazy-load the selected model")
    yield
    log.info("🛑 Shutting down")
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
    clear_models()
    cache_service.clear()


app = FastAPI(title="DepthLens Pro API", version="3.1.0", debug=settings.DEBUG, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.exception_handler(Exception)
async def _err(req: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled: %s", req.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
