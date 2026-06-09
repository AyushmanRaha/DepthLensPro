"""FastAPI route definitions for DepthLens Pro."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.config import settings
from backend.api.live import SERVICE_VERSION


def _available_devices():
    from backend.utils.hardware import _available_devices as impl
    return impl()


def _default_device_key():
    from backend.utils.hardware import _default_device_key as impl
    return impl()


def _acceleration_checks(devs):
    from backend.utils.hardware import _acceleration_checks as impl
    return impl(devs)


def _resolve(device):
    from backend.utils.hardware import _resolve as impl
    return impl(device)


def _inference():
    from backend.services import inference
    return inference


def _cache_service():
    from backend.services import cache_service
    return cache_service


def run_benchmark(model: str, device: str, iterations: int):
    from backend.services.benchmarks import run_benchmark as impl
    return impl(model=model, device=device, iterations=iterations)


def process_image(*args, **kwargs):
    return _inference().process_image(*args, **kwargs)


def loaded_model_keys():
    try:
        return _inference().loaded_model_keys()
    except Exception as exc:
        log.warning("Loaded model inspection degraded: %s", exc)
        return []


def _torch_version() -> str | None:
    try:
        return __import__("torch").__version__
    except Exception as exc:
        log.warning("Torch version inspection degraded: %s", exc)
        return None


def _fhash(*args, **kwargs):
    return _inference()._fhash(*args, **kwargs)


def normalize_metrics_mode(*args, **kwargs):
    return _inference().normalize_metrics_mode(*args, **kwargs)


def parse_outputs(*args, **kwargs):
    return _inference().parse_outputs(*args, **kwargs)

log = logging.getLogger("depthlens")
router = APIRouter()

MEMORY_PRESSURE_LIMIT_PERCENT = 90.0
DISK_USAGE_LIMIT_PERCENT = 90.0
DISK_TELEMETRY_PATH = "/"
_PROBE_TTL_SECONDS = 10.0
_DEVICE_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "devices": None,
    "primary": "cpu",
    "error": None,
}
_ACCEL_CACHE: dict[str, Any] = {"expires_at": 0.0, "checks": None, "error": None}
_ROUTE_INFERENCE_SEMAPHORE = asyncio.Semaphore(
    max(1, int(os.getenv("INFERENCE_MAX_CONCURRENCY", "2")))
)


async def process_image_async(
    raw: bytes,
    model: str,
    colormap: str,
    device: str,
    filename: str | None,
    metrics: str | None = None,
    outputs: str | None = None,
    max_dim: int | None = None,
) -> dict[str, Any]:
    """Offload blocking image inference while preserving route-level monkeypatching."""

    async with _ROUTE_INFERENCE_SEMAPHORE:
        return await run_in_threadpool(
            process_image, raw, model, colormap, device, filename, metrics, outputs, max_dim
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _fallback_cpu(error: str | None = None) -> dict[str, Any]:
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "System CPU"
    return {
        "cpu": {
            "name": f"CPU · {cpu_name}",
            "hardware_name": cpu_name,
            "type": "cpu",
            "compute_classes": ["cpu"],
            "available": True,
            **({"discovery_error": error} if error else {}),
        }
    }


def _cached_devices(force: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
    now = time.time()
    if (
        not force
        and _DEVICE_CACHE.get("devices") is not None
        and float(_DEVICE_CACHE.get("expires_at", 0.0)) > now
    ):
        return (
            _DEVICE_CACHE["devices"],
            str(_DEVICE_CACHE.get("primary") or "cpu"),
            {
                "cached": True,
                "error": _DEVICE_CACHE.get("error"),
            },
        )

    started = time.perf_counter()
    error = None
    try:
        devs = _available_devices()
        if "cpu" not in devs:
            devs = {**_fallback_cpu("device discovery omitted CPU"), **devs}
        primary = _default_device_key()
        if primary not in devs:
            primary = "cpu"
    except Exception as exc:
        error = str(exc)
        log.warning("Device discovery degraded: %s", exc)
        devs = _fallback_cpu(error)
        primary = "cpu"

    _DEVICE_CACHE.update(
        {
            "expires_at": now + _PROBE_TTL_SECONDS,
            "devices": devs,
            "primary": primary,
            "error": error,
        }
    )
    return devs, primary, {"cached": False, "error": error, "duration_ms": _elapsed_ms(started)}


def _cached_acceleration_checks(
    devs: dict[str, Any], force: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = time.time()
    if (
        not force
        and _ACCEL_CACHE.get("checks") is not None
        and float(_ACCEL_CACHE.get("expires_at", 0.0)) > now
    ):
        return _ACCEL_CACHE["checks"], {"cached": True, "error": _ACCEL_CACHE.get("error")}

    started = time.perf_counter()
    error = None
    try:
        checks = _acceleration_checks(devs)
    except Exception as exc:
        error = str(exc)
        log.warning("Acceleration probe degraded: %s", exc)
        checks = {
            "cuda": {
                "available": any(k.startswith("cuda:") for k in devs),
                "operational": False,
                "error": error,
            },
            "mps": {"available": "mps" in devs, "operational": False, "error": error},
            "xpu": {
                "available": any(k.startswith("xpu:") for k in devs),
                "operational": False,
                "error": error,
            },
        }
    _ACCEL_CACHE.update({"expires_at": now + _PROBE_TTL_SECONDS, "checks": checks, "error": error})
    return checks, {"cached": False, "error": error, "duration_ms": _elapsed_ms(started)}


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


@router.get("/health")
async def health() -> dict[str, Any]:
    started = time.perf_counter()
    devs, best, device_meta = _cached_devices()
    accel = {k: v for k, v in devs.items() if k != "cpu"}
    checks, accel_meta = _cached_acceleration_checks(devs)
    accel_ok = (
        True
        if not accel
        else any(c.get("operational") for c in checks.values() if c.get("available"))
    )

    memory = _memory_telemetry()
    disk = _disk_telemetry()
    cache_started = time.perf_counter()
    try:
        cache_metrics = _cache_service().metrics()
        cache_error = None
    except Exception as exc:
        log.warning("Cache metrics degraded: %s", exc)
        cache_metrics = {"status": "degraded", "error": str(exc)}
        cache_error = str(exc)
    cache_ms = _elapsed_ms(cache_started)

    status = _telemetry_status(memory, disk)
    if device_meta.get("error") or accel_meta.get("error") or cache_error:
        status = "degraded"

    return {
        "status": status,
        "diagnostics_status": status,
        "version": SERVICE_VERSION,
        "primary_device": best,
        "devices": devs,
        "loaded_models": loaded_model_keys(),
        "cache_entries": _cache_service().size(),
        "cache_metrics": cache_metrics,
        "torch_version": _torch_version(),
        "cuda_available": any(k.startswith("cuda:") for k in devs),
        "mps_available": "mps" in devs,
        "xpu_available": any(k.startswith("xpu:") for k in devs),
        "acceleration_ok": accel_ok,
        "acceleration_checks": checks,
        "warmup": {
            "enabled": settings.DEPTHLENS_PRELOAD_MODEL,
            "model": settings.DEPTHLENS_WARMUP_MODEL,
            "device": settings.DEPTHLENS_WARMUP_DEVICE,
            "loaded_models": loaded_model_keys(),
        },
        "timings_ms": {
            "device_discovery": device_meta.get("duration_ms", 0.0),
            "accelerator_probe": accel_meta.get("duration_ms", 0.0),
            "cache_metrics": cache_ms,
            "health_generation": _elapsed_ms(started),
        },
        "telemetry": {
            "memory": memory,
            "disk": disk,
        },
        "system": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "cpu": devs.get("cpu", {}).get("hardware_name", "System CPU"),
            "accelerators": [d["name"] for d in accel.values() if "name" in d],
        },
    }


@router.get("/devices")
async def list_devices() -> dict[str, Any]:
    devs, primary, meta = _cached_devices()
    return {"devices": devs, "primary_device": primary, "cached": meta.get("cached", False)}


@router.get("/models")
async def list_models() -> dict[str, list[dict[str, str]]]:
    return {"models": [{"id": k, **v} for k, v in _inference().SUPPORTED_MODELS.items()]}


@router.get("/colormaps")
async def list_colormaps() -> dict[str, list[str]]:
    return {"colormaps": list(_inference().COLORMAPS.keys())}


@router.get("/api/benchmark")
@router.get("/benchmark")
async def benchmark(
    model: str = "MiDaS_small", device: str = "auto", iterations: int = 3
) -> dict[str, Any]:
    """Return PyTorch and ONNX Runtime performance matrices for the UI."""

    try:
        return await run_in_threadpool(
            run_benchmark, model=model, device=device, iterations=iterations
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/estimate")
async def estimate(
    file: UploadFile = File(...),
    model: str = Form("MiDaS_small"),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
    metrics: str = Form(settings.DEPTHLENS_DEFAULT_METRICS),
    outputs: str = Form(settings.DEPTHLENS_DEFAULT_OUTPUTS),
    max_dim: int | None = Form(None),
) -> JSONResponse:
    if model not in _inference().SUPPORTED_MODELS:
        raise HTTPException(422, f"Unknown model '{model}'")
    if colormap not in _inference().COLORMAPS:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")
    try:
        metrics = normalize_metrics_mode(metrics)
        outputs = ",".join(parse_outputs(outputs))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    devs, _, _ = _cached_devices()
    avail = list(devs.keys()) + ["auto"]
    if device not in avail:
        raise HTTPException(422, f"Device '{device}' unavailable. Options: {avail}")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")

    raw = await file.read()
    if len(raw) / 1024**2 > _inference().MAX_SIZE_MB:
        raise HTTPException(413, f"File exceeds {_inference().MAX_SIZE_MB} MB limit")

    resolved = str(_resolve(device))
    ck = _fhash(raw, model, colormap, resolved, metrics, outputs, max_dim)
    cached = _cache_service().get(ck)
    if cached is not None:
        log.info("Cache hit: %r", file.filename)
        return JSONResponse({**cached, "cached": True})

    try:
        result = await process_image_async(
            raw, model, colormap, resolved, file.filename, metrics, outputs, max_dim
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {exc}") from exc

    _cache_service().set(ck, result)
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
    metrics: str = Form(settings.DEPTHLENS_DEFAULT_METRICS),
    outputs: str = Form(settings.DEPTHLENS_DEFAULT_OUTPUTS),
    max_dim: int | None = Form(None),
) -> JSONResponse:
    if len(files) > 10:
        raise HTTPException(422, "Batch limit: 10 images")
    if model not in _inference().SUPPORTED_MODELS:
        raise HTTPException(422, f"Unknown model '{model}'")
    if colormap not in _inference().COLORMAPS:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")
    try:
        metrics = normalize_metrics_mode(metrics)
        outputs = ",".join(parse_outputs(outputs))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    resolved = str(_resolve(device))
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str | None]] = []
    for upload in files:
        try:
            raw = await upload.read()
            if len(raw) / 1024**2 > _inference().MAX_SIZE_MB:
                raise ValueError("File too large")
            ck = _fhash(raw, model, colormap, resolved, metrics, outputs, max_dim)
            cached = _cache_service().get(ck)
            if cached is not None:
                results.append({**cached, "cached": True})
                continue

            res = await process_image_async(
                raw, model, colormap, resolved, upload.filename, metrics, outputs, max_dim
            )
            _cache_service().set(ck, res)
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

    return _cache_service().metrics()


@router.delete("/cache")
async def clear_cache() -> dict[str, int]:
    return {"cleared": _cache_service().clear()}
