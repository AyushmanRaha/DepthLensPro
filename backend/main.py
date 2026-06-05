"""FastAPI application factory and runtime lifecycle."""

from __future__ import annotations

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
from backend.services.inference import _load_model, clear_models
from backend.utils.hardware import _available_devices, _default_device_key


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the default model on startup and clear runtime caches on shutdown."""
    devs = _available_devices()
    best = _default_device_key()
    log.info("🚀 DepthLens Pro v3.1 — devices: %s  best: %s", list(devs), best)
    try:
        _load_model("MiDaS_small", best)
        log.info("✅ MiDaS_small pre-loaded on %s", best)
    except Exception as exc:
        log.warning("⚠️  Pre-load on %s skipped: %s", best, exc)
        try:
            _load_model("MiDaS_small", "cpu")
            log.info("✅ MiDaS_small pre-loaded on CPU (fallback)")
        except Exception as fallback_exc:
            log.warning("⚠️  CPU pre-load also skipped: %s", fallback_exc)
    yield
    log.info("🛑 Shutting down")
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
