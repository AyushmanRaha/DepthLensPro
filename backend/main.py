"""FastAPI application factory and runtime lifecycle."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.services import cache_service
from backend.services.inference import _load_model, clear_models
from backend.utils.hardware import _available_devices, _default_device_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
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


app = FastAPI(title="DepthLens Pro API", version="3.1.0", lifespan=lifespan)
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
