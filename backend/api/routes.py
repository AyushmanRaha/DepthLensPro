"""FastAPI route definitions for DepthLens Pro."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import time
from typing import Any, cast

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.api.live import SERVICE_VERSION
from backend.config import settings
from backend.model_metadata import COLORMAP_NAMES
from backend.model_registry import UnknownModelError, normalize_model_id, supported_models_payload


def _available_devices() -> dict[str, Any]:
    from backend.utils.hardware import _available_devices as impl

    return impl()


def _default_device_key() -> str:
    from backend.utils.hardware import _default_device_key as impl

    return impl()


def _acceleration_checks(devs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from backend.utils.hardware import _acceleration_checks as impl

    return impl(devs)


def _resolve(device: str) -> Any:
    from backend.utils.hardware import _resolve as impl

    return impl(device)


def _inference() -> Any:
    from backend.services import inference

    return inference


def _cache_service() -> Any:
    from backend.services import cache_service

    return cache_service


def run_benchmark(model: str, device: str, iterations: int) -> dict[str, Any]:
    from backend.services.benchmarks import run_benchmark as impl

    return impl(model=model, device=device, iterations=iterations)


def onnx_status_payload(device: str = "auto") -> dict[str, Any]:
    from backend.services.onnx_diagnostics import onnx_status_payload as impl

    return impl(device=device)


def readiness_payload(device: str = "auto") -> dict[str, Any]:
    from backend.services.onnx_diagnostics import readiness_payload as impl

    return impl(device)


def process_image(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _inference().process_image(*args, **kwargs))


def reconstruct_point_cloud(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from backend.services.reconstruction import reconstruct_point_cloud as impl

    return impl(*args, **kwargs)


def loaded_model_keys() -> list[str]:
    try:
        return cast(list[str], _inference().loaded_model_keys())
    except Exception as exc:
        log.warning("Loaded model inspection degraded: %s", exc)
        return []


def _torch_version() -> str | None:
    try:
        return cast(str, __import__("torch").__version__)
    except Exception as exc:
        log.warning("Torch version inspection degraded: %s", exc)
        return None


def _fhash(*args: Any, **kwargs: Any) -> str:
    return cast(str, _inference()._fhash(*args, **kwargs))


def normalize_metrics_mode(*args: Any, **kwargs: Any) -> str:
    return cast(str, _inference().normalize_metrics_mode(*args, **kwargs))


def parse_outputs(*args: Any, **kwargs: Any) -> list[str]:
    return cast(list[str], _inference().parse_outputs(*args, **kwargs))


log = logging.getLogger("depthlens")
router = APIRouter()


def _dependency_unavailable(exc: Exception) -> HTTPException:
    log.warning("Inference runtime dependency unavailable: %s: %s", type(exc).__name__, exc)
    return HTTPException(
        503,
        {
            "error_code": "INFERENCE_RUNTIME_UNAVAILABLE",
            "message": "Inference runtime is not ready. Check /ready for dependency diagnostics.",
        },
    )


MEMORY_PRESSURE_LIMIT_PERCENT = 90.0
MAX_UPLOAD_SIZE_MB = 20
BENCHMARK_TIMEOUT_MESSAGE = "Benchmark timed out · depth engine remains available"
DISK_USAGE_LIMIT_PERCENT = 90.0
DISK_TELEMETRY_PATH = "/"
_PROBE_TTL_SECONDS = 10.0
_DEVICE_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "devices": None,
    "primary": "cpu",
    "error": None,
}
_ACCEL_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "checks": None,
    "error": None,
    "device_keys": (),
}
_READINESS_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "device": None,
    "payload": None,
    "error": None,
}


async def process_image_async(
    raw: bytes,
    model: str,
    colormap: str,
    device: str,
    filename: str | None,
    metrics: str | None = None,
    outputs: str | None = None,
    max_dim: int | None = None,
    gt_raw: bytes | None = None,
    gt_filename: str | None = None,
    gt_required: bool = False,
    gt_scale: float | None = None,
    gt_invalid_value: float | None = None,
) -> dict[str, Any]:
    """Offload blocking image inference while preserving route-level monkeypatching."""

    # This route wrapper is the single API request offload point.  It deliberately
    # does not use the service-level async helper, so INFERENCE_MAX_CONCURRENCY=2
    # is not accidentally applied twice for HTTP requests.
    return await run_in_threadpool(
        process_image,
        raw,
        model,
        colormap,
        device,
        filename,
        metrics,
        outputs,
        max_dim,
        gt_raw,
        gt_filename,
        gt_required,
        gt_scale,
        gt_invalid_value,
    )


async def reconstruct_point_cloud_async(
    *,
    raw: bytes,
    filename: str | None,
    model: str,
    device: str,
    colormap: str = "inferno",
    max_dim: int | None = None,
    export_format: str = "ply",
    max_points: int = 120000,
    preview_points: int = 5000,
    focal_scale: float = 1.2,
    depth_scale: float = 1.0,
    depth_near_percentile: float = 2.0,
    depth_far_percentile: float = 98.0,
    sampling: str = "grid",
    include_rgb: bool = True,
    coordinate_system: str = "y_up",
) -> dict[str, Any]:
    """Offload blocking point-cloud reconstruction while preserving route monkeypatching."""

    return await run_in_threadpool(
        reconstruct_point_cloud,
        raw=raw,
        filename=filename,
        model=model,
        device=device,
        colormap=colormap,
        max_dim=max_dim,
        export_format=export_format,
        max_points=max_points,
        preview_points=preview_points,
        focal_scale=focal_scale,
        depth_scale=depth_scale,
        depth_near_percentile=depth_near_percentile,
        depth_far_percentile=depth_far_percentile,
        sampling=sampling,
        include_rgb=include_rgb,
        coordinate_system=coordinate_system,
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
    device_keys = tuple(sorted(devs))
    if (
        not force
        and _ACCEL_CACHE.get("checks") is not None
        and _ACCEL_CACHE.get("device_keys") == device_keys
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
    _ACCEL_CACHE.update(
        {
            "expires_at": now + _PROBE_TTL_SECONDS,
            "checks": checks,
            "error": error,
            "device_keys": device_keys,
        }
    )
    return checks, {"cached": False, "error": error, "duration_ms": _elapsed_ms(started)}


def _cached_readiness_payload(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cache heavier ONNX/model readiness diagnostics for health polling."""

    now = time.time()
    if (
        _READINESS_CACHE.get("payload") is not None
        and _READINESS_CACHE.get("device") == device
        and float(_READINESS_CACHE.get("expires_at", 0.0)) > now
    ):
        return _READINESS_CACHE["payload"], {"cached": True, "error": _READINESS_CACHE.get("error")}

    started = time.perf_counter()
    error = None
    try:
        payload = readiness_payload(device)
    except Exception as exc:
        error = str(exc)
        log.warning("Readiness diagnostics degraded: %s", exc)
        payload = {
            "status": "degraded",
            "overall_status": "diagnostics_unavailable",
            "error": error,
        }
    _READINESS_CACHE.update(
        {
            "expires_at": now + _PROBE_TTL_SECONDS,
            "device": device,
            "payload": payload,
            "error": error,
        }
    )
    return payload, {"cached": False, "error": error, "duration_ms": _elapsed_ms(started)}


def _validated_device_or_422(device: str) -> str:
    """Resolve a requested device with one stale-cache refresh before returning 422."""

    def available_options(force: bool = False) -> list[str]:
        devs, _, _ = _cached_devices(force=force)
        return [*devs.keys(), "auto"]

    avail = available_options()
    if device not in avail:
        raise HTTPException(422, f"Device '{device}' is unavailable. Options: {avail}")
    try:
        return str(_resolve(device))
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"device": device})
        raise HTTPException(
            504,
            {
                "error_code": "BENCHMARK_TIMEOUT",
                "message": BENCHMARK_TIMEOUT_MESSAGE,
            },
        ) from exc
    except ValueError as exc:
        refreshed = available_options(force=True)
        if device not in refreshed:
            detail = f"Device '{device}' is unavailable. Options: {refreshed}"
            raise HTTPException(422, detail) from exc
        try:
            return str(_resolve(device))
        except ValueError as refreshed_exc:
            raise HTTPException(422, str(refreshed_exc)) from refreshed_exc


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
    try:
        onnx = onnx_status_payload(best)
        onnx_error = None
    except Exception as exc:
        log.warning("ONNX diagnostics degraded: %s", exc)
        onnx = {"status": "degraded", "error": str(exc)}
        onnx_error = str(exc)
    readiness, readiness_meta = _cached_readiness_payload(best)

    status = _telemetry_status(memory, disk)
    if (
        device_meta.get("error")
        or accel_meta.get("error")
        or cache_error
        or onnx_error
        or readiness_meta.get("error")
    ):
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
        "onnx": onnx,
        "readiness": readiness,
        "backend_live": True,
        "overall_status": readiness.get("overall_status"),
        "model_readiness": readiness.get("models", {}),
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
            "readiness": readiness_meta.get("duration_ms", 0.0),
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


@router.get("/ready")
async def ready() -> dict[str, Any]:
    """Report whether required inference dependencies are importable without loading models."""

    from backend.services.diagnostics import readiness_payload

    return await run_in_threadpool(readiness_payload)


@router.get("/devices")
async def list_devices() -> dict[str, Any]:
    devs, primary, meta = _cached_devices()
    return {"devices": devs, "primary_device": primary, "cached": meta.get("cached", False)}


@router.get("/onnx/status")
async def onnx_status(device: str = "auto") -> dict[str, Any]:
    """Expose static ONNX weight and runtime provider diagnostics."""

    return await run_in_threadpool(onnx_status_payload, device=device)


@router.get("/models")
async def list_models() -> dict[str, Any]:
    return {"models": supported_models_payload()}


@router.get("/colormaps")
async def list_colormaps() -> dict[str, list[str]]:
    return {"colormaps": list(COLORMAP_NAMES)}


@router.get("/api/benchmark")
@router.get("/benchmark")
async def benchmark(
    model: str = "MiDaS_small", device: str = "auto", iterations: int = 3
) -> dict[str, Any]:
    """Return PyTorch and ONNX Runtime performance matrices for the UI."""

    try:
        from backend.services.benchmarks import BENCHMARK_TIMEOUT_SECONDS

        return await asyncio.wait_for(
            run_in_threadpool(run_benchmark, model=model, device=device, iterations=iterations),
            timeout=BENCHMARK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise HTTPException(
            504,
            {
                "error_code": "BENCHMARK_TIMEOUT",
                "message": BENCHMARK_TIMEOUT_MESSAGE,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Benchmark runtime unavailable")
        raise _dependency_unavailable(exc) from exc


@router.post("/estimate")
async def estimate(
    file: UploadFile = File(...),
    model: str = Form("MiDaS_small"),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
    metrics: str = Form(settings.DEPTHLENS_DEFAULT_METRICS),
    outputs: str = Form(settings.DEPTHLENS_DEFAULT_OUTPUTS),
    max_dim: int | None = Form(None),
    gt_file: UploadFile | None = File(None),
    gt_required: bool = Form(False),
    gt_scale: float | None = Form(None),
    gt_invalid_value: float | None = Form(None),
) -> JSONResponse:
    try:
        model = normalize_model_id(model)
    except UnknownModelError as exc:
        raise HTTPException(
            422,
            {"error_code": exc.error_code, "message": str(exc), "valid_models": exc.valid_models},
        ) from exc
    if colormap not in COLORMAP_NAMES:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")
    try:
        metrics = normalize_metrics_mode(metrics)
        outputs = ",".join(parse_outputs(outputs))
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise HTTPException(
            504,
            {
                "error_code": "BENCHMARK_TIMEOUT",
                "message": BENCHMARK_TIMEOUT_MESSAGE,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise _dependency_unavailable(exc) from exc

    resolved = _validated_device_or_422(device)

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image file exceeds {MAX_UPLOAD_SIZE_MB} MB limit")

    gt_raw: bytes | None = None
    gt_filename: str | None = None
    if gt_file is not None:
        gt_raw = await gt_file.read()
        gt_filename = gt_file.filename
        if len(gt_raw) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"GT depth file exceeds {MAX_UPLOAD_SIZE_MB} MB limit")
    elif gt_required:
        raise HTTPException(422, "GT mode requires one image and one GT depth file")

    # GT metrics depend on uploaded labels and must not reuse image-only cached payloads.
    ck = (
        None
        if gt_raw is not None or gt_required
        else _fhash(raw, model, colormap, resolved, metrics, outputs, max_dim)
    )
    cached = _cache_service().get(ck) if ck is not None else None
    if cached is not None:
        log.info("Cache hit: %r", file.filename)
        return JSONResponse({**cached, "cached": True})

    try:
        result = await process_image_async(
            raw,
            model,
            colormap,
            resolved,
            file.filename,
            metrics,
            outputs,
            max_dim,
            gt_raw,
            gt_filename,
            gt_required,
            gt_scale,
            gt_invalid_value,
        )
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise HTTPException(
            504,
            {
                "error_code": "BENCHMARK_TIMEOUT",
                "message": BENCHMARK_TIMEOUT_MESSAGE,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(
            500,
            {"error_code": "INFERENCE_FAILED", "message": "Inference failed"},
        ) from exc

    if ck is not None:
        _cache_service().set(ck, result)
    log.info(
        "✅ %r | %s | %s | %s ms",
        file.filename,
        model,
        resolved,
        result["latency_ms"],
    )
    return JSONResponse(result)


@router.post("/api/reconstruct")
@router.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(...),
    model: str = Form("MiDaS_small"),
    device: str = Form("auto"),
    colormap: str = Form("inferno"),
    max_dim: int | None = Form(None),
    export_format: str = Form("ply"),
    max_points: int = Form(120000),
    preview_points: int = Form(5000),
    focal_scale: float = Form(1.2),
    depth_scale: float = Form(1.0),
    depth_near_percentile: float = Form(2.0),
    depth_far_percentile: float = Form(98.0),
    sampling: str = Form("grid"),
    include_rgb: bool = Form(True),
    coordinate_system: str = Form("y_up"),
) -> JSONResponse:
    try:
        model = normalize_model_id(model)
    except UnknownModelError as exc:
        raise HTTPException(
            422,
            {"error_code": exc.error_code, "message": str(exc), "valid_models": exc.valid_models},
        ) from exc
    if colormap not in COLORMAP_NAMES:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")

    resolved = _validated_device_or_422(device)

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image file exceeds {MAX_UPLOAD_SIZE_MB} MB limit")

    try:
        result = await reconstruct_point_cloud_async(
            raw=raw,
            filename=file.filename,
            model=model,
            device=resolved,
            colormap=colormap,
            max_dim=max_dim,
            export_format=export_format,
            max_points=max_points,
            preview_points=preview_points,
            focal_scale=focal_scale,
            depth_scale=depth_scale,
            depth_near_percentile=depth_near_percentile,
            depth_far_percentile=depth_far_percentile,
            sampling=sampling,
            include_rgb=include_rgb,
            coordinate_system=coordinate_system,
        )
    except asyncio.TimeoutError as exc:
        log.warning("Reconstruction timed out", extra={"model": model, "device": device})
        raise HTTPException(
            504,
            {
                "error_code": "RECONSTRUCTION_TIMEOUT",
                "message": "Point cloud generation timed out · depth engine remains available",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Reconstruction failed")
        raise HTTPException(
            500,
            {"error_code": "RECONSTRUCTION_FAILED", "message": "Point cloud generation failed"},
        ) from exc

    log.info(
        "✅ reconstruct %r | %s | %s | %s pts | %s | %s ms",
        file.filename,
        model,
        resolved,
        result.get("reconstruction", {}).get("point_count"),
        result.get("artifact_format"),
        result.get("total_latency_ms"),
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
        raise HTTPException(422, "Batch limit is 10 images")
    try:
        model = normalize_model_id(model)
    except UnknownModelError as exc:
        raise HTTPException(
            422,
            {"error_code": exc.error_code, "message": str(exc), "valid_models": exc.valid_models},
        ) from exc
    if colormap not in COLORMAP_NAMES:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")
    try:
        metrics = normalize_metrics_mode(metrics)
        outputs = ",".join(parse_outputs(outputs))
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise HTTPException(
            504,
            {
                "error_code": "BENCHMARK_TIMEOUT",
                "message": BENCHMARK_TIMEOUT_MESSAGE,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise _dependency_unavailable(exc) from exc
    resolved = _validated_device_or_422(device)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str | None]] = []
    for upload in files:
        try:
            if not (upload.content_type or "").startswith("image/"):
                raise ValueError("Expected an image file")
            raw = await upload.read()
            if len(raw) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise ValueError(f"Image file exceeds {MAX_UPLOAD_SIZE_MB} MB limit")
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

    return cast(dict[str, Any], _cache_service().metrics())


@router.delete("/cache")
async def clear_cache() -> dict[str, int]:
    return {"cleared": int(_cache_service().clear())}
