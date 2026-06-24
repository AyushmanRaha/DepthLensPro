# FastAPI middleware and exception decorators are dynamically typed in the installed stubs.
# mypy: disable-error-code=untyped-decorator

"""FastAPI application factory and runtime lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.live import SERVICE_VERSION
from backend.api.live import router as live_router
from backend.api.routes import router
from backend.config import settings
from backend.services import observability


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
    if os.getenv("DEPTHLENS_SKIP_WARMUP") == "1" or os.getenv("TESTING") == "1":
        log.info("Model warmup skipped by DEPTHLENS_SKIP_WARMUP/TESTING")
        return

    requested_model = settings.DEPTHLENS_WARMUP_MODEL
    requested_device = settings.DEPTHLENS_WARMUP_DEVICE
    from backend.model_registry import UnknownModelError, normalize_model_id
    from backend.services.inference import _load_model
    from backend.utils.hardware import _resolve

    try:
        model = normalize_model_id(requested_model)
    except UnknownModelError:
        log.warning("⚠️  Warmup skipped: unknown model %s", requested_model)
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
        # Reuse the normal loading path and locks so background warmup cannot race
        # a first request into duplicate torch.hub.load calls for the same key.
        await asyncio.to_thread(_load_model, model, device)
        log.info("✅ Background warmup complete: %s on %s", model, device)
    except Exception as exc:
        log.warning("⚠️  Background warmup failed for %s on %s: %s", model, device, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start quickly; optional model warmup runs after liveness is available."""
    log.info("ASGI_STARTUP_BEGIN")
    warmup_task: asyncio.Task[None] | None = None
    if settings.DEPTHLENS_PRELOAD_MODEL:
        warmup_task = asyncio.create_task(_warm_default_model())
    else:
        log.info("Model preload disabled; first inference will lazy-load the selected model")
    log.info("ASGI_STARTUP_YIELDING")
    log.info("ASGI_STARTUP_COMPLETE")
    try:
        yield
    finally:
        log.info("🛑 Shutting down")
        if warmup_task and not warmup_task.done():
            warmup_task.cancel()
        try:
            from backend.services.inference import clear_models

            clear_models()
        except Exception as exc:
            log.warning("Model cleanup degraded: %s", exc)
        try:
            from backend.services import cache_service

            cache_service.clear()
        except Exception as exc:
            log.warning("Cache cleanup degraded: %s", exc)


app = FastAPI(
    title="DepthLens Pro API", version=SERVICE_VERSION, debug=settings.DEBUG, lifespan=lifespan
)


def _cors_origins() -> list[str]:
    if settings.DEPTHLENS_CORS_ALLOW_ALL:
        return ["*"]
    origins = {
        "null",
        "file://",
        "http://localhost",
        "http://127.0.0.1",
        f"http://localhost:{settings.PORT}",
        f"http://127.0.0.1:{settings.PORT}",
    }
    origins.update(
        item.strip() for item in settings.DEPTHLENS_CORS_ALLOWED_ORIGINS.split(",") if item.strip()
    )
    return sorted(origins)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):[0-9]{1,5}$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(live_router)
app.include_router(router)


@app.middleware("http")
async def _observability_middleware(request: Request, call_next: Any) -> Any:
    started = time.perf_counter()
    observability.increment_active_http()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(getattr(response, "status_code", 500))
        return response
    finally:
        route_obj = request.scope.get("route")
        route_path = getattr(route_obj, "path", None) or request.url.path
        observability.observe_http_request(
            request.method, str(route_path), status_code, time.perf_counter() - started
        )
        observability.decrement_active_http()


@app.exception_handler(Exception)
async def _err(req: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled: %s", observability.sanitize_message(req.url))
    observability.record_crash("api", "UNHANDLED_EXCEPTION", exc, route=str(req.url.path))
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "Internal server error",
                "retryable": True,
            }
        },
    )
