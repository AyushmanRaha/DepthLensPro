"""FastAPI route definitions for DepthLens Pro."""

from __future__ import annotations

import logging
import platform
from typing import Any

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.services import cache_service
from backend.services.inference import (
    COLORMAPS,
    MAX_SIZE_MB,
    SUPPORTED_MODELS,
    _fhash,
    loaded_model_keys,
    process_image,
)
from backend.utils.hardware import (
    _acceleration_checks,
    _available_devices,
    _default_device_key,
    _resolve,
)

log = logging.getLogger("depthlens")
router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    return {"service": "DepthLens Pro API", "version": "3.1.0"}


@router.get("/health")
async def health() -> dict[str, Any]:
    devs = _available_devices()
    best = _default_device_key()
    accel = {k: v for k, v in devs.items() if k != "cpu"}
    checks = _acceleration_checks(devs)
    accel_ok = any(c.get("operational") for c in checks.values() if c.get("available"))

    return {
        "status": "ok",
        "version": "3.1.0",
        "primary_device": best,
        "devices": devs,
        "loaded_models": loaded_model_keys(),
        "cache_entries": cache_service.size(),
        "torch_version": torch.__version__,
        "cuda_available": any(k.startswith("cuda:") for k in devs),
        "mps_available": "mps" in devs,
        "xpu_available": any(k.startswith("xpu:") for k in devs),
        "acceleration_ok": bool(accel) and accel_ok,
        "acceleration_checks": checks,
        "system": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "cpu": devs["cpu"]["hardware_name"],
            "accelerators": [d["name"] for d in accel.values()],
        },
    }


@router.get("/devices")
async def list_devices() -> dict[str, Any]:
    return {"devices": _available_devices()}


@router.get("/models")
async def list_models() -> dict[str, list[dict[str, str]]]:
    return {"models": [{"id": k, **v} for k, v in SUPPORTED_MODELS.items()]}


@router.get("/colormaps")
async def list_colormaps() -> dict[str, list[str]]:
    return {"colormaps": list(COLORMAPS.keys())}


@router.post("/estimate")
async def estimate(
    file: UploadFile = File(...),
    model: str = Form("MiDaS_small"),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
) -> JSONResponse:
    if model not in SUPPORTED_MODELS:
        raise HTTPException(422, f"Unknown model '{model}'")
    if colormap not in COLORMAPS:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")

    avail = list(_available_devices().keys()) + ["auto"]
    if device not in avail:
        raise HTTPException(422, f"Device '{device}' unavailable. Options: {avail}")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")

    raw = await file.read()
    if len(raw) / 1024**2 > MAX_SIZE_MB:
        raise HTTPException(413, f"File exceeds {MAX_SIZE_MB} MB limit")

    resolved = str(_resolve(device))
    ck = _fhash(raw, model, colormap, resolved)
    cached = cache_service.get(ck)
    if cached is not None:
        log.info("Cache hit: %r", file.filename)
        return JSONResponse({**cached, "cached": True})

    try:
        result = process_image(raw, model, colormap, resolved, file.filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {exc}") from exc

    cache_service.set(ck, result)
    log.info(
        "✅ %r | %s | %s | %s ms",
        file.filename,
        model,
        resolved,
        result["latency_ms"],
    )
    return JSONResponse(result)


@router.post("/batch")
async def batch(
    files: list[UploadFile] = File(...),
    model: str = Form("MiDaS_small"),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
) -> JSONResponse:
    if len(files) > 10:
        raise HTTPException(422, "Batch limit: 10 images")
    resolved = str(_resolve(device))
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str | None]] = []
    for upload in files:
        try:
            raw = await upload.read()
            if len(raw) / 1024**2 > MAX_SIZE_MB:
                raise ValueError("File too large")
            ck = _fhash(raw, model, colormap, resolved)
            cached = cache_service.get(ck)
            if cached is not None:
                results.append({**cached, "cached": True})
                continue

            res = process_image(raw, model, colormap, resolved, upload.filename)
            cache_service.set(ck, res)
            results.append(res)
        except Exception as exc:
            errors.append({"filename": upload.filename, "error": str(exc)})
    return JSONResponse(
        {
            "results": results,
            "errors": errors,
            "total": len(files),
            "succeeded": len(results),
            "failed": len(errors),
        }
    )


@router.delete("/cache")
async def clear_cache() -> dict[str, int]:
    return {"cleared": cache_service.clear()}
