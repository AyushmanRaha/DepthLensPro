"""FastAPI route definitions for DepthLens Pro."""

from __future__ import annotations

import logging
import os
import platform
import shutil
from typing import Any

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.services import cache_service
from backend.services.benchmarks import run_benchmark
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

MEMORY_PRESSURE_LIMIT_PERCENT = 90.0
DISK_USAGE_LIMIT_PERCENT = 90.0
DISK_TELEMETRY_PATH = "/"


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _memory_telemetry() -> dict[str, Any]:
    """Return local memory pressure telemetry without external system dependencies."""

    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
            for line in meminfo_file:
                key, raw_value = line.split(":", 1)
                meminfo[key] = int(raw_value.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        meminfo = {}

    if meminfo.get("MemTotal") and meminfo.get("MemAvailable") is not None:
        total_bytes = meminfo["MemTotal"]
        available_bytes = meminfo["MemAvailable"]
    elif hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            total_bytes = page_size * physical_pages
            available_bytes = page_size * available_pages
        except (ValueError, OSError, AttributeError):
            return {
                "status": "unknown",
                "pressure_percent": None,
                "limit_percent": MEMORY_PRESSURE_LIMIT_PERCENT,
            }
    else:
        return {
            "status": "unknown",
            "pressure_percent": None,
            "limit_percent": MEMORY_PRESSURE_LIMIT_PERCENT,
        }

    used_bytes = max(total_bytes - available_bytes, 0)
    pressure_percent = _percent(used_bytes, total_bytes)
    return {
        "status": "ok" if pressure_percent < MEMORY_PRESSURE_LIMIT_PERCENT else "degraded",
        "pressure_percent": pressure_percent,
        "limit_percent": MEMORY_PRESSURE_LIMIT_PERCENT,
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "used_bytes": used_bytes,
    }


def _disk_telemetry(path: str = DISK_TELEMETRY_PATH) -> dict[str, Any]:
    """Return local disk utilization telemetry for the supplied filesystem path."""

    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {
            "status": "unknown",
            "path": path,
            "usage_percent": None,
            "limit_percent": DISK_USAGE_LIMIT_PERCENT,
            "error": str(exc),
        }

    used_bytes = usage.total - usage.free
    usage_percent = _percent(used_bytes, usage.total)
    return {
        "status": "ok" if usage_percent < DISK_USAGE_LIMIT_PERCENT else "degraded",
        "path": path,
        "usage_percent": usage_percent,
        "limit_percent": DISK_USAGE_LIMIT_PERCENT,
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "used_bytes": used_bytes,
    }


def _telemetry_status(*checks: dict[str, Any]) -> str:
    return "degraded" if any(check.get("status") == "degraded" for check in checks) else "ok"


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

    memory = _memory_telemetry()
    disk = _disk_telemetry()

    return {
        "status": _telemetry_status(memory, disk),
        "version": "3.1.0",
        "primary_device": best,
        "devices": devs,
        "loaded_models": loaded_model_keys(),
        "cache_entries": cache_service.size(),
        "cache_metrics": cache_service.metrics(),
        "torch_version": torch.__version__,
        "cuda_available": any(k.startswith("cuda:") for k in devs),
        "mps_available": "mps" in devs,
        "xpu_available": any(k.startswith("xpu:") for k in devs),
        "acceleration_ok": bool(accel) and accel_ok,
        "acceleration_checks": checks,
        "telemetry": {
            "memory": memory,
            "disk": disk,
        },
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


@router.get("/api/benchmark")
@router.get("/benchmark")
async def benchmark(
    model: str = "MiDaS_small", device: str = "auto", iterations: int = 3
) -> dict[str, Any]:
    """Return PyTorch and ONNX Runtime performance matrices for the UI."""

    try:
        return run_benchmark(model=model, device=device, iterations=iterations)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


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


@router.get("/cache/metrics")
async def cache_metrics() -> dict[str, Any]:
    """Expose live Redis/fallback cache metrics for frontend dashboards."""

    return cache_service.metrics()


@router.delete("/cache")
async def clear_cache() -> dict[str, int]:
    return {"cleared": cache_service.clear()}
