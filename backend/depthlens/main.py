import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router
from .config import APP_NAME, APP_VERSION
from .devices import available_devices
from .models import load_model
from .runtime import CACHE, MODELS, TRANSFORMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("depthlens")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("🚀 DepthLens Pro — devices: %s", list(available_devices().keys()))
    try:
        load_model("MiDaS_small", "cpu")
        logger.info("✅ MiDaS_small pre-loaded on CPU")
    except Exception as exc:
        logger.warning("⚠️ Pre-load skipped: %s", exc)
    yield
    logger.info("🛑 Shutting down")
    MODELS.clear()
    TRANSFORMS.clear()
    CACHE.clear()


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled error for %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
