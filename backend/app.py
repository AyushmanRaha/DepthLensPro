"""
DepthLens Pro — Backend API
FastAPI + PyTorch MiDaS depth estimation server
"""

import io
import time
import base64
import logging
import hashlib
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("depthlens")

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    "MiDaS_small":  {"label": "Small",  "description": "Fastest — ~30 ms"},
    "DPT_Hybrid":   {"label": "Hybrid", "description": "Balanced — ~120 ms"},
    "DPT_Large":    {"label": "Large",  "description": "Highest accuracy — ~400 ms"},
}

COLORMAPS = {
    "inferno":  cv2.COLORMAP_INFERNO,
    "plasma":   cv2.COLORMAP_PLASMA,
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "magma":    cv2.COLORMAP_MAGMA,
    "jet":      cv2.COLORMAP_JET,
    "hot":      cv2.COLORMAP_HOT,
    "bone":     cv2.COLORMAP_BONE,
    "turbo":    cv2.COLORMAP_TURBO,
}

MAX_IMAGE_DIMENSION = 2048
MAX_FILE_SIZE_MB    = 20
RESULT_CACHE: dict  = {}          # simple in-process cache {hash: result}
MODELS: dict        = {}          # loaded model instances
TRANSFORMS: dict    = {}          # corresponding transforms


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 DepthLens Pro server starting …")
    # Pre-load the lightweight model so the first request isn't cold
    try:
        _load_model("MiDaS_small")
        log.info("✅ MiDaS_small pre-loaded successfully")
    except Exception as exc:
        log.warning(f"⚠️  Could not pre-load MiDaS_small: {exc}")
    yield
    log.info("🛑 DepthLens Pro server shutting down")
    MODELS.clear()
    TRANSFORMS.clear()
    RESULT_CACHE.clear()


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DepthLens Pro API",
    description="Monocular depth estimation with MiDaS model family",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_model(model_name: str):
    """Load and cache a MiDaS model + its transform."""
    if model_name in MODELS:
        return MODELS[model_name], TRANSFORMS[model_name]

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")

    log.info(f"Loading model '{model_name}' from Torch Hub …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        "intel-isl/MiDaS", model_name, trust_repo=True
    )
    model.to(device).eval()

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True
    )
    transform = (
        midas_transforms.small_transform
        if model_name == "MiDaS_small"
        else midas_transforms.dpt_transform
    )

    MODELS[model_name]    = (model, device)
    TRANSFORMS[model_name] = transform
    log.info(f"✅ Model '{model_name}' ready on {device}")
    return (model, device), transform


def _file_hash(data: bytes, model: str, colormap: str) -> str:
    key = f"{model}:{colormap}:{hashlib.md5(data).hexdigest()}"
    return hashlib.sha1(key.encode()).hexdigest()


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes to an OpenCV BGR image, with size validation."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file")

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        log.debug(f"Resized from ({w}×{h}) → ({img.shape[1]}×{img.shape[0]})")
    return img


def _run_inference(img_bgr: np.ndarray, model_name: str) -> np.ndarray:
    """Run MiDaS inference; returns normalised float32 depth map [0..1]."""
    (model, device), transform = _load_model(model_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)
    return depth.astype(np.float32)


def _apply_colormap(depth: np.ndarray, colormap_name: str) -> np.ndarray:
    cmap_id = COLORMAPS.get(colormap_name, cv2.COLORMAP_INFERNO)
    depth_uint8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cmap_id)


def _to_base64_png(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _depth_statistics(depth: np.ndarray) -> dict:
    """Return a small stats dict from the normalised depth map."""
    flat = depth.flatten()
    hist, edges = np.histogram(flat, bins=32, range=(0.0, 1.0))
    return {
        "min":      round(float(flat.min()), 4),
        "max":      round(float(flat.max()), 4),
        "mean":     round(float(flat.mean()), 4),
        "std":      round(float(flat.std()), 4),
        "median":   round(float(np.median(flat)), 4),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": [round(e, 3) for e in edges.tolist()],
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def root():
    return {
        "service": "DepthLens Pro API",
        "version": "2.0.0",
        "status":  "operational",
    }


@app.get("/health", tags=["health"])
async def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = list(MODELS.keys())
    return {
        "status":         "ok",
        "device":         device,
        "loaded_models":  loaded,
        "cache_entries":  len(RESULT_CACHE),
        "torch_version":  torch.__version__,
    }


@app.get("/models", tags=["models"])
async def list_models():
    return {
        "models": [
            {"id": k, **v} for k, v in SUPPORTED_MODELS.items()
        ]
    }


@app.get("/colormaps", tags=["config"])
async def list_colormaps():
    return {"colormaps": list(COLORMAPS.keys())}


@app.post("/estimate", tags=["inference"])
async def estimate_depth(
    file:     UploadFile = File(...),
    model:    str        = Form("MiDaS_small"),
    colormap: str        = Form("inferno"),
):
    """
    Estimate depth for a single uploaded image.

    Returns a JSON object containing:
    - `depth_map`   — base64-encoded PNG (colourised)
    - `grayscale`   — base64-encoded PNG (raw grayscale)
    - `stats`       — depth statistics dict
    - `latency_ms`  — server-side inference time
    - `model`       — model name used
    - `colormap`    — colormap name used
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Valid: {list(SUPPORTED_MODELS)}"
        )
    if colormap not in COLORMAPS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown colormap '{colormap}'. Valid: {list(COLORMAPS)}"
        )

    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Upload an image."
        )

    raw = await file.read()
    size_mb = len(raw) / (1024 ** 2)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Limit: {MAX_FILE_SIZE_MB} MB."
        )

    # ── Cache look-up ────────────────────────────────────────────────────────
    cache_key = _file_hash(raw, model, colormap)
    if cache_key in RESULT_CACHE:
        log.info(f"Cache hit for {file.filename!r}")
        return JSONResponse(content={**RESULT_CACHE[cache_key], "cached": True})

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        img_bgr = _decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    t0 = time.perf_counter()
    try:
        depth = _run_inference(img_bgr, model)
    except Exception as exc:
        log.exception(f"Inference failed for '{file.filename}'")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Post-processing ───────────────────────────────────────────────────────
    colourised   = _apply_colormap(depth, colormap)
    depth_uint8  = (depth * 255).astype(np.uint8)
    gray_bgr     = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    stats = _depth_statistics(depth)

    result = {
        "depth_map":  _to_base64_png(colourised),
        "grayscale":  _to_base64_png(gray_bgr),
        "stats":      stats,
        "latency_ms": latency_ms,
        "model":      model,
        "colormap":   colormap,
        "resolution": {"width": img_bgr.shape[1], "height": img_bgr.shape[0]},
        "filename":   file.filename,
        "cached":     False,
    }

    RESULT_CACHE[cache_key] = result
    log.info(
        f"✅ '{file.filename}' | {model} | {colormap} | "
        f"{img_bgr.shape[1]}×{img_bgr.shape[0]} | {latency_ms} ms"
    )
    return JSONResponse(content=result)


@app.post("/batch", tags=["inference"])
async def batch_estimate(
    files:    list[UploadFile] = File(...),
    model:    str              = Form("MiDaS_small"),
    colormap: str              = Form("inferno"),
):
    """Process multiple images in a single request."""
    if len(files) > 10:
        raise HTTPException(
            status_code=422,
            detail="Batch limit is 10 images per request."
        )

    results = []
    errors  = []

    for f in files:
        try:
            raw = await f.read()
            size_mb = len(raw) / (1024 ** 2)
            if size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"File too large ({size_mb:.1f} MB)")

            cache_key = _file_hash(raw, model, colormap)
            if cache_key in RESULT_CACHE:
                results.append({**RESULT_CACHE[cache_key], "cached": True})
                continue

            img_bgr  = _decode_image(raw)
            t0       = time.perf_counter()
            depth    = _run_inference(img_bgr, model)
            latency  = round((time.perf_counter() - t0) * 1000, 1)

            colourised  = _apply_colormap(depth, colormap)
            depth_uint8 = (depth * 255).astype(np.uint8)
            gray_bgr    = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

            res = {
                "depth_map":  _to_base64_png(colourised),
                "grayscale":  _to_base64_png(gray_bgr),
                "stats":      _depth_statistics(depth),
                "latency_ms": latency,
                "model":      model,
                "colormap":   colormap,
                "resolution": {"width": img_bgr.shape[1], "height": img_bgr.shape[0]},
                "filename":   f.filename,
                "cached":     False,
            }
            RESULT_CACHE[cache_key] = res
            results.append(res)

        except Exception as exc:
            log.warning(f"Batch item '{f.filename}' failed: {exc}")
            errors.append({"filename": f.filename, "error": str(exc)})

    return JSONResponse(content={
        "results":      results,
        "errors":       errors,
        "total":        len(files),
        "succeeded":    len(results),
        "failed":       len(errors),
    })


@app.delete("/cache", tags=["admin"])
async def clear_cache():
    count = len(RESULT_CACHE)
    RESULT_CACHE.clear()
    return {"cleared": count, "message": f"Removed {count} cached result(s)."}


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled error on {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )