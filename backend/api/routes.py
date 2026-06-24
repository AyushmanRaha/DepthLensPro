# FastAPI's route decorators are dynamically typed in the installed stubs.
# mypy: disable-error-code=untyped-decorator

"""FastAPI route definitions for DepthLens Pro."""

from __future__ import annotations

import asyncio
import logging
import platform
import time
from typing import Any, cast

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from backend.api import device_state
from backend.api.errors import (
    benchmark_timeout as benchmark_timeout_error,
)
from backend.api.errors import (
    detector_unavailable,
    embedded_error,
    generic_detector_failure,
    generic_inference_failure,
    generic_reconstruction_failure,
    inference_dependency_unavailable,
    model_assets_unavailable,
    timeout_error,
    validation_error,
)
from backend.api.errors import (
    reconstruction_timeout as reconstruction_timeout_error,
)
from backend.api.live import SERVICE_VERSION
from backend.api.system_telemetry import disk_telemetry, memory_telemetry, telemetry_status
from backend.api.validation import (
    normalize_request_colormap,
    normalize_request_metrics_and_outputs,
    normalize_request_model,
    validate_detection_params,
    validate_gt_upload,
    validate_image_upload_content_type,
    validate_max_dim,
    validate_reconstruction_params,
    validate_upload_size,
)
from backend.config import settings
from backend.constants import MAX_UPLOAD_SIZE_MB
from backend.model_metadata import COLORMAP_NAMES
from backend.model_registry import MODEL_REGISTRY, supported_models_payload
from backend.services import observability
from backend.services.model_assets import ModelAssetsUnavailableError


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


def onnx_status_payload(
    device: str = "auto", *, depth: str = "quick", force: bool = False
) -> dict[str, Any]:
    from backend.services.onnx_diagnostics import onnx_status_payload as impl

    return impl(device=device, depth=depth, force=force)


def readiness_payload(
    device: str = "auto", *, depth: str = "quick", force: bool = False
) -> dict[str, Any]:
    from backend.services.onnx_diagnostics import readiness_payload as impl

    return impl(device, depth=depth, force=force)


def process_image(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _inference().process_image(*args, **kwargs))


def reconstruct_point_cloud(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from backend.services.reconstruction import reconstruct_point_cloud as impl

    return impl(*args, **kwargs)


def detect_objects(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from backend.services.object_detection import detect_objects as impl

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


async def _with_route_timeout(
    awaitable: Any,
    route: str,
    model: str | None = None,
    device: str | None = None,
    timeout_factory: Any = None,
    timeout_code: str = "REQUEST_TIMEOUT",
) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=settings.DEPTHLENS_ROUTE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        observability.record_crash("api", timeout_code, exc, route=route)
        if model and device:
            observability.record_inference(
                model, "unknown", device, None, outcome="error", error_code=timeout_code
            )
        log.warning(
            "Route timed out", extra={"route": route, "model": model or "", "device": device or ""}
        )
        raise (timeout_factory() if timeout_factory else timeout_error(route)) from exc


async def _with_batch_item_timeout(awaitable: Any, route: str = "/batch") -> Any:
    try:
        return await asyncio.wait_for(
            awaitable, timeout=settings.DEPTHLENS_BATCH_ITEM_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        observability.record_crash("api", "REQUEST_TIMEOUT", exc, route=route)
        raise timeout_error(route) from exc


def _normalize_compare_models(models: str | None) -> list[str]:
    raw_models = (
        list(MODEL_REGISTRY)
        if models is None or not models.strip()
        else [item.strip() for item in models.split(",")]
    )
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_model in raw_models:
        if not raw_model:
            continue
        model_id = normalize_request_model(raw_model)
        if model_id not in seen:
            normalized.append(model_id)
            seen.add(model_id)
    if not normalized:
        raise HTTPException(
            422,
            {
                "error_code": "INVALID_MODEL",
                "message": "At least one model is required",
                "field": "models",
            },
        )
    return normalized


def _latency_value(result: dict[str, Any]) -> float | None:
    value = result.get("latency_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _comparison_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    timed = [
        (result.get("model_id") or result.get("model"), latency)
        for result in results
        if (latency := _latency_value(result)) is not None
    ]
    if not timed:
        return {
            "fastest_model_id": None,
            "lowest_latency_ms": None,
            "slowest_model_id": None,
            "highest_latency_ms": None,
        }
    fastest = min(timed, key=lambda item: item[1])
    slowest = max(timed, key=lambda item: item[1])
    return {
        "fastest_model_id": fastest[0],
        "lowest_latency_ms": fastest[1],
        "slowest_model_id": slowest[0],
        "highest_latency_ms": slowest[1],
    }


def _safe_compare_error(model: str, exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "error_code", exc.__class__.__name__)
    if isinstance(exc, ModelAssetsUnavailableError):
        message = "Model assets unavailable"
    elif isinstance(exc, ValueError):
        message = str(exc)
    else:
        message = "Depth inference failed"
    detail = embedded_error(str(code), message)
    return {
        "model_id": model,
        "error_code": detail["error_code"],
        "message": detail["message"],
        "error": detail["message"],
        "error_detail": detail,
    }


def _dependency_unavailable(exc: Exception) -> HTTPException:
    return inference_dependency_unavailable(exc, log)


_DEVICE_CACHE = device_state.DEVICE_CACHE
_ACCEL_CACHE = device_state.ACCEL_CACHE
_READINESS_CACHE = device_state.READINESS_CACHE


def _elapsed_ms(start: float) -> float:
    return device_state.elapsed_ms(start)


def _fallback_cpu(error: str | None = None) -> dict[str, Any]:
    return device_state.fallback_cpu(error)


def _cached_devices(force: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
    device_state.DEVICE_CACHE = _DEVICE_CACHE
    return device_state.cached_devices(
        available_devices=_available_devices,
        default_device_key=_default_device_key,
        log=log,
        force=force,
    )


def _cached_acceleration_checks(
    devs: dict[str, Any], force: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    device_state.ACCEL_CACHE = _ACCEL_CACHE
    return device_state.cached_acceleration_checks(
        devs, acceleration_checks=_acceleration_checks, log=log, force=force
    )


def _cached_readiness_payload(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    device_state.READINESS_CACHE = _READINESS_CACHE
    return device_state.cached_readiness_payload(
        device, readiness_payload=readiness_payload, log=log
    )


def _validated_device_or_422(device: str) -> str:
    return device_state.validated_device_or_422(
        device, cached_devices_func=_cached_devices, resolve=_resolve, log=log
    )


def _memory_telemetry() -> dict[str, Any]:
    return memory_telemetry()


def _disk_telemetry() -> dict[str, Any]:
    return disk_telemetry()


def _telemetry_status(*checks: dict[str, Any]) -> str:
    return telemetry_status(*checks)


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
    return cast(
        dict[str, Any],
        await run_in_threadpool(
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
        ),
    )


async def detect_objects_async(**kwargs: Any) -> dict[str, Any]:
    """Offload blocking local object detection while preserving route monkeypatching."""

    return cast(dict[str, Any], await run_in_threadpool(detect_objects, **kwargs))


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

    return cast(
        dict[str, Any],
        await run_in_threadpool(
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
        ),
    )


def _required_device_health_failed(devs: dict[str, Any]) -> bool:
    cpu = devs.get("cpu")
    return not isinstance(cpu, dict) or cpu.get("available") is not True


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    content, media_type = observability.prometheus_text()
    return Response(content=content, media_type=media_type)


@router.get("/api/observability")
@router.get("/observability")
async def observability_snapshot() -> dict[str, Any]:
    return observability.snapshot()


@router.get("/health")
async def health(depth: str = "quick", force: bool = False) -> dict[str, Any]:
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
        onnx = onnx_status_payload(best, depth=depth, force=force)
    except Exception as exc:
        log.warning("ONNX diagnostics degraded: %s", exc)
        onnx = {"status": "degraded", "error": str(exc)}
    readiness, readiness_meta = _cached_readiness_payload(best)

    status = _telemetry_status(memory, disk)
    if cache_error or _required_device_health_failed(devs):
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
async def ready(depth: str = "quick", force: bool = False) -> dict[str, Any]:
    """Report whether required inference dependencies are importable without loading models."""

    from backend.services.diagnostics import readiness_payload

    return cast(
        dict[str, Any], await run_in_threadpool(readiness_payload, depth=depth, force=force)
    )


@router.get("/devices")
async def list_devices() -> dict[str, Any]:
    devs, primary, meta = _cached_devices()
    return {"devices": devs, "primary_device": primary, "cached": meta.get("cached", False)}


@router.get("/onnx/status")
async def onnx_status(
    device: str = "auto", depth: str = "quick", force: bool = False
) -> dict[str, Any]:
    """Expose static ONNX weight and runtime provider diagnostics."""

    return cast(
        dict[str, Any],
        await run_in_threadpool(onnx_status_payload, device=device, depth=depth, force=force),
    )


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

        result: dict[str, Any] = await asyncio.wait_for(
            run_in_threadpool(run_benchmark, model=model, device=device, iterations=iterations),
            timeout=BENCHMARK_TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError as exc:
        observability.record_benchmark(
            model, None, device, device, iterations, None, None, None, None, None, 1, "error"
        )
        log.warning("Route timed out", extra={"model": model, "device": device})
        raise benchmark_timeout_error() from exc
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except Exception as exc:
        observability.record_crash("benchmark", "BENCHMARK_FAILED", exc, route="/api/benchmark")
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
    model = normalize_request_model(model)
    colormap = normalize_request_colormap(colormap)
    validate_max_dim(max_dim)
    try:
        metrics, outputs = normalize_request_metrics_and_outputs(
            metrics,
            outputs,
            normalize_metrics_mode=normalize_metrics_mode,
            parse_outputs=parse_outputs,
        )
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise benchmark_timeout_error() from exc
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except Exception as exc:
        raise _dependency_unavailable(exc) from exc

    resolved = _validated_device_or_422(device)

    validate_image_upload_content_type(file)

    raw = await file.read()
    validate_upload_size(raw, max_size_mb=MAX_UPLOAD_SIZE_MB)

    gt_raw: bytes | None = None
    gt_filename: str | None = None
    if gt_file is not None:
        gt_raw = await gt_file.read()
        gt_filename = gt_file.filename
        validate_gt_upload(gt_file, gt_raw)
    elif gt_required:
        raise HTTPException(
            422,
            {
                "error_code": "MISSING_GT",
                "message": "GT mode requires one image and one GT depth file",
                "field": "gt_file",
            },
        )

    # GT metrics depend on uploaded labels and must not reuse image-only cached payloads.
    ck = (
        None
        if gt_raw is not None or gt_required
        else _fhash(raw, model, colormap, resolved, metrics, outputs, max_dim)
    )
    output_set = parse_outputs(outputs)
    cached = _cache_service().get(ck) if ck is not None else None
    if ck is not None:
        observability.safe_observe(
            "cache_event",
            observability.record_cache_event,
            "hit" if cached is not None else "miss",
            "route",
        )
    if cached is not None:
        log.info("Cache hit for uploaded image")
        observability.safe_observe(
            "route_cache_hit_inference",
            observability.record_inference,
            model,
            cached.get("engine_used", "cache"),
            resolved,
            cached.get("latency_ms"),
            cached=True,
            outcome="ok",
            metrics_mode=metrics,
            outputs_count=len(output_set),
        )
        return JSONResponse({**cached, "cached": True})

    try:
        result = await _with_route_timeout(
            process_image_async(
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
            ),
            "/estimate",
            model,
            resolved,
        )
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise benchmark_timeout_error() from exc
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except ModelAssetsUnavailableError as exc:
        observability.record_crash("inference", exc.error_code, exc, route="/estimate")
        observability.record_inference(
            model,
            "unknown",
            resolved,
            None,
            cached=False,
            outcome="error",
            error_code=exc.error_code,
        )
        raise model_assets_unavailable(exc) from exc
    except Exception as exc:
        observability.record_crash("inference", "INFERENCE_FAILED", exc, route="/estimate")
        observability.record_inference(
            model,
            "unknown",
            resolved,
            None,
            cached=False,
            outcome="error",
            error_code="INFERENCE_FAILED",
        )
        log.exception("Inference failed")
        raise generic_inference_failure() from exc

    if ck is not None:
        _cache_service().set(ck, result)
        observability.safe_observe("cache_event", observability.record_cache_event, "set", "route")
    observability.safe_observe(
        "route_inference",
        observability.record_inference,
        model,
        result.get("engine_used", "pytorch"),
        resolved,
        result.get("latency_ms"),
        pixels=(
            int(result.get("resolution", {}).get("width", 0))
            * int(result.get("resolution", {}).get("height", 0))
        ),
        cached=False,
        outcome="ok",
        metrics_mode=metrics,
        outputs_count=len(output_set),
    )
    log.info(
        "estimate completed | %s | %s | %s ms",
        model,
        resolved,
        result["latency_ms"],
    )
    return JSONResponse(result)


@router.post("/api/compare")
@router.post("/compare")
async def compare(
    file: UploadFile = File(...),
    models: str | None = Form(None),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
    metrics: str = Form("full"),
    outputs: str = Form("color,gray"),
    max_dim: int | None = Form(None),
) -> JSONResponse:
    model_ids = _normalize_compare_models(models)
    colormap = normalize_request_colormap(colormap)
    validate_max_dim(max_dim)
    try:
        metrics, outputs = normalize_request_metrics_and_outputs(
            metrics,
            outputs,
            normalize_metrics_mode=normalize_metrics_mode,
            parse_outputs=parse_outputs,
        )
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except Exception as exc:
        raise _dependency_unavailable(exc) from exc

    resolved = _validated_device_or_422(device)
    validate_image_upload_content_type(file)
    raw = await file.read()
    validate_upload_size(raw, max_size_mb=MAX_UPLOAD_SIZE_MB)

    output_set = parse_outputs(outputs)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with observability.trace_span(
        "api",
        "compare",
        {
            "device_type": observability.normalize_device_type(resolved),
            "model_count": len(model_ids),
        },
    ):
        for model_id in model_ids:
            try:
                ck = _fhash(raw, model_id, colormap, resolved, metrics, outputs, max_dim)
                cached = _cache_service().get(ck)
                observability.safe_observe(
                    "cache_event",
                    observability.record_cache_event,
                    "hit" if cached is not None else "miss",
                    "route",
                )
                if cached is not None:
                    result = {**cached, "cached": True}
                    results.append(result)
                    observability.safe_observe(
                        "compare_cache_hit_inference",
                        observability.record_inference,
                        model_id,
                        cached.get("engine_used", "cache"),
                        resolved,
                        cached.get("latency_ms"),
                        cached=True,
                        outcome="ok",
                        metrics_mode=metrics,
                        outputs_count=len(output_set),
                    )
                    continue

                result = await _with_route_timeout(
                    process_image_async(
                        raw, model_id, colormap, resolved, file.filename, metrics, outputs, max_dim
                    ),
                    "/compare",
                    model_id,
                    resolved,
                )
                result.setdefault("model_id", model_id)
                _cache_service().set(ck, result)
                observability.safe_observe(
                    "cache_event", observability.record_cache_event, "set", "route"
                )
                results.append(result)
                observability.safe_observe(
                    "compare_inference",
                    observability.record_inference,
                    model_id,
                    result.get("engine_used", "pytorch"),
                    resolved,
                    result.get("latency_ms"),
                    pixels=(
                        int(result.get("resolution", {}).get("width", 0))
                        * int(result.get("resolution", {}).get("height", 0))
                    ),
                    cached=False,
                    outcome="ok",
                    metrics_mode=metrics,
                    outputs_count=len(output_set),
                )
            except Exception as exc:
                observability.record_crash(
                    "inference", "COMPARE_ITEM_FAILED", exc, route="/compare"
                )
                observability.record_inference(
                    model_id,
                    "unknown",
                    resolved,
                    None,
                    cached=False,
                    outcome="error",
                    error_code=getattr(exc, "error_code", "COMPARE_ITEM_FAILED"),
                )
                errors.append(_safe_compare_error(model_id, exc))

    return JSONResponse(
        {
            "filename": file.filename,
            "device_used": resolved,
            "models": model_ids,
            "results": results,
            "errors": errors,
            "total": len(model_ids),
            "succeeded": len(results),
            "failed": len(errors),
            "comparison": _comparison_summary(results),
        }
    )


@router.post("/api/detect")
@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    threshold: float = Form(0.35),
    max_detections: int = Form(5),
) -> JSONResponse:
    validate_image_upload_content_type(file)
    validate_detection_params(threshold, max_detections)
    resolved = _validated_device_or_422(device)
    raw = await file.read()
    validate_upload_size(raw, max_size_mb=MAX_UPLOAD_SIZE_MB)
    try:
        result = await _with_route_timeout(
            detect_objects_async(
                raw=raw,
                filename=file.filename,
                device=resolved,
                threshold=threshold,
                max_detections=max_detections,
            ),
            "/detect",
            "object_detection",
            resolved,
        )
        observability.record_inference(
            "object_detection",
            "detector",
            resolved,
            result.get("latency_ms") or result.get("total_latency_ms"),
            outcome="ok",
        )
        return JSONResponse(result)
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except Exception as exc:
        if (
            getattr(exc, "error_code", None) == "DETECTOR_WEIGHTS_UNAVAILABLE"
            or exc.__class__.__name__ == "DetectorUnavailableError"
            or (
                isinstance(exc, ModuleNotFoundError) and exc.name in {"PIL", "torch", "torchvision"}
            )
        ):
            raise detector_unavailable(exc) from exc
        observability.record_crash("detector", "DETECTION_FAILED", exc, route="/detect")
        observability.record_inference(
            "object_detection",
            "detector",
            resolved,
            None,
            outcome="error",
            error_code="DETECTION_FAILED",
        )
        log.exception("Object detection failed")
        raise generic_detector_failure() from exc


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
    model = normalize_request_model(model)
    colormap = normalize_request_colormap(colormap)
    validate_max_dim(max_dim)
    validate_reconstruction_params(
        max_points,
        preview_points,
        focal_scale,
        depth_scale,
        depth_near_percentile,
        depth_far_percentile,
        sampling,
        coordinate_system,
    )

    resolved = _validated_device_or_422(device)

    validate_image_upload_content_type(file)

    raw = await file.read()
    validate_upload_size(raw, max_size_mb=MAX_UPLOAD_SIZE_MB)

    try:
        result = await _with_route_timeout(
            reconstruct_point_cloud_async(
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
            ),
            "/reconstruct",
            model,
            resolved,
            timeout_factory=reconstruction_timeout_error,
            timeout_code="RECONSTRUCTION_TIMEOUT",
        )
    except asyncio.TimeoutError as exc:
        log.warning("Reconstruction timed out", extra={"model": model, "device": device})
        raise reconstruction_timeout_error() from exc
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except ModelAssetsUnavailableError as exc:
        observability.record_crash("reconstruction", exc.error_code, exc, route="/reconstruct")
        observability.record_inference(
            model, "reconstruction", resolved, None, outcome="error", error_code=exc.error_code
        )
        raise model_assets_unavailable(exc) from exc
    except Exception as exc:
        observability.record_crash(
            "reconstruction", "RECONSTRUCTION_FAILED", exc, route="/reconstruct"
        )
        observability.record_inference(
            model,
            "reconstruction",
            resolved,
            None,
            outcome="error",
            error_code="RECONSTRUCTION_FAILED",
        )
        log.exception("Reconstruction failed")
        raise generic_reconstruction_failure() from exc

    observability.record_inference(
        model,
        result.get("engine_used", "reconstruction"),
        resolved,
        result.get("latency_ms") or result.get("total_latency_ms"),
        outcome="ok",
    )
    log.info(
        "reconstruct completed | %s | %s | %s pts | %s | %s ms",
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
        raise validation_error("Batch limit is 10 images", field="files")
    model = normalize_request_model(model)
    colormap = normalize_request_colormap(colormap)
    validate_max_dim(max_dim)
    try:
        metrics, outputs = normalize_request_metrics_and_outputs(
            metrics,
            outputs,
            normalize_metrics_mode=normalize_metrics_mode,
            parse_outputs=parse_outputs,
        )
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"model": model, "device": device})
        raise benchmark_timeout_error() from exc
    except ValueError as exc:
        raise validation_error(str(exc)) from exc
    except Exception as exc:
        raise _dependency_unavailable(exc) from exc
    resolved = _validated_device_or_422(device)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with observability.trace_span(
        "api",
        "batch",
        {"model_id": model, "device_type": observability.normalize_device_type(resolved)},
    ):
        for upload in files:
            try:
                validate_image_upload_content_type(upload)
                raw = await upload.read()
                validate_upload_size(raw, max_size_mb=MAX_UPLOAD_SIZE_MB)
                ck = _fhash(raw, model, colormap, resolved, metrics, outputs, max_dim)
                cached = _cache_service().get(ck)
                if cached is not None:
                    observability.record_cache_event("hit", "route")
                    observability.record_inference(
                        model,
                        cached.get("engine_used", "cache"),
                        resolved,
                        cached.get("latency_ms"),
                        cached=True,
                        outcome="ok",
                        metrics_mode=metrics,
                    )
                    results.append({**cached, "cached": True})
                    continue
                observability.record_cache_event("miss", "route")

                res = await _with_batch_item_timeout(
                    process_image_async(
                        raw, model, colormap, resolved, upload.filename, metrics, outputs, max_dim
                    )
                )
                _cache_service().set(ck, res)
                observability.record_cache_event("set", "route")
                results.append(res)
            except Exception as exc:
                observability.record_crash("inference", "BATCH_ITEM_FAILED", exc, route="/batch")
                observability.record_inference(
                    model,
                    "unknown",
                    resolved,
                    None,
                    outcome="error",
                    error_code="BATCH_ITEM_FAILED",
                )
                detail = embedded_error(
                    getattr(exc, "error_code", "BATCH_ITEM_FAILED"),
                    str(getattr(exc, "detail", exc)),
                )
                errors.append(
                    {
                        "filename": upload.filename,
                        "error": detail["message"],
                        "error_code": detail["error_code"],
                        "error_detail": detail,
                    }
                )
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

    data = cast(dict[str, Any], _cache_service().metrics())
    observability.record_cache_event(
        "metrics", str(data.get("backend", "unknown")), data.get("keyspace_size")
    )
    return data


@router.delete("/cache")
async def clear_cache() -> dict[str, int]:
    cleared = int(_cache_service().clear())
    observability.record_cache_event("clear", "route", 0)
    return {"cleared": cleared}


@router.post("/cache/clear")
async def clear_cache_post() -> dict[str, int]:
    """Clear the active inference cache from browser clients that prefer POST."""

    cleared = int(_cache_service().clear())
    observability.record_cache_event("clear", "route", 0)
    return {"cleared": cleared}
