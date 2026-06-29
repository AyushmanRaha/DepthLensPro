"""Depth-estimation inference façade and orchestration services."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, cast

import cv2
import numpy as np
from starlette.concurrency import run_in_threadpool

from backend.constants import MAX_UPLOAD_SIZE_MB
from backend.depth_models import ONNXExecutionEngine
from backend.model_metadata import SUPPORTED_MODELS
from backend.model_registry import get_model_spec, normalize_model_id, resolve_onnx_path
from backend.services import observability
from backend.services.engine_selector import normalize_engine_mode, select_engine_for_inference
from backend.services.ground_truth import (
    GroundTruthError,
    compute_ground_truth_metrics,
    decode_ground_truth,
)
from backend.services.image_io import COLORMAPS, _b64, _colorize, _decode, _normalize_depth
from backend.services.inference_cache import (
    _DEPTH_CACHE,
    _DEPTH_CACHE_LOCK,
    _depth_cache_key,
    _fhash,
    _get_cached_depth,
    _raw_hash,
    _set_cached_depth,
)
from backend.services.metrics import (
    _compute_fast_metrics,
    _compute_metrics,
    _metrics_for_mode,
    _ssim,
    _with_metric_groups,
    normalize_metrics_mode,
    parse_outputs,
)
from backend.services.model_runtime import (
    _MODEL_FORWARD_LOCKS,
    _MODEL_LOCK,
    MODELS,
    TRANSFORMS,
    _infer_torch,
    _load_model,
)
from backend.services.onnx_inference import (
    _ONNX_FORWARD_LOCKS,
    _ONNX_LOCK,
    _ONNX_MISSING_WARNED,
    ONNX_ENGINES,
)

__all__ = [
    "COLORMAPS",
    "MAX_SIZE_MB",
    "SUPPORTED_MODELS",
    "process_image",
    "process_image_async",
    "infer_depth_arrays",
    "clear_models",
    "loaded_model_keys",
    "normalize_metrics_mode",
    "parse_outputs",
    "_fhash",
    "_raw_hash",
    "_depth_cache_key",
    "_get_cached_depth",
    "_set_cached_depth",
    "_decode",
    "_normalize_depth",
    "_colorize",
    "_b64",
    "_compute_fast_metrics",
    "_compute_metrics",
    "_metrics_for_mode",
    "_ssim",
    "_with_metric_groups",
    "_load_model",
    "_infer_torch",
    "_infer_onnx",
    "_infer_with_metadata",
    "_load_onnx_engine",
    "_load_onnx_engine_with_lock",
    "resolve_onnx_path",
    "ONNXExecutionEngine",
]

MAX_SIZE_MB = MAX_UPLOAD_SIZE_MB
_INFERENCE_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("INFERENCE_MAX_CONCURRENCY", "2"))))


def clear_models() -> None:
    """Release all loaded model and transform references."""
    with _MODEL_LOCK, _ONNX_LOCK:
        MODELS.clear()
        ONNX_ENGINES.clear()
        TRANSFORMS.clear()
        _MODEL_FORWARD_LOCKS.clear()
        _ONNX_FORWARD_LOCKS.clear()
        with _DEPTH_CACHE_LOCK:
            _DEPTH_CACHE.clear()
        _ONNX_MISSING_WARNED.clear()


def loaded_model_keys() -> list[str]:
    """Return cache keys for models currently loaded in memory."""
    return list(MODELS.keys()) + [f"onnx:{key}" for key in ONNX_ENGINES]


def _load_onnx_engine_with_lock(
    model_name: str, device_str: str
) -> tuple[ONNXExecutionEngine, Any]:
    """Compatibility wrapper for ONNX engine loading and forward locks."""

    model_id = normalize_model_id(model_name)
    key = f"{model_id}:{device_str}"
    with _ONNX_LOCK:
        forward_lock = _ONNX_FORWARD_LOCKS.setdefault(key, __import__("threading").Lock())
        engine = ONNX_ENGINES.get(key)
        if engine is None:
            engine = ONNXExecutionEngine(model_name=model_id, device=device_str)
            ONNX_ENGINES[key] = engine
        return engine, forward_lock


def _load_onnx_engine(model_name: str, device_str: str) -> ONNXExecutionEngine:
    """Load or reuse an ONNX Runtime execution engine for static MiDaS weights."""

    engine, _forward_lock = _load_onnx_engine_with_lock(model_name, device_str)
    return engine


def _infer_onnx(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Run normalized ONNX Runtime depth inference for a BGR image."""

    from backend.services.image_io import _normalize_depth

    model_id = normalize_model_id(model_name)
    engine, forward_lock = _load_onnx_engine_with_lock(model_id, device_str)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with forward_lock:
        depth = engine.forward(rgb)
    return _normalize_depth(depth)


def _infer_with_metadata(
    img_bgr: np.ndarray, model_name: str, device_str: str, engine_requested: str = "auto"
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility wrapper preserving monkeypatchable fallback hooks."""

    model_id = normalize_model_id(model_name)
    spec = get_model_spec(model_id)
    requested = normalize_engine_mode(engine_requested)
    warnings: list[str] = []
    resolved = resolve_onnx_path(model_id)
    selection = select_engine_for_inference(model_id, device_str, requested, resolved)
    selected_engine = str(selection.get("selected_engine") or selection.get("engine") or "pytorch")
    fallback_reason: str | None = None

    if selected_engine == "onnxruntime":
        try:
            depth = _infer_onnx(img_bgr, model_id, device_str)
            provider = getattr(_load_onnx_engine(model_id, device_str), "provider", "onnxruntime")
            return depth, {
                "model_id": model_id,
                "model_display_name": spec.display_name,
                "engine_requested": requested,
                "engine_used": "onnxruntime",
                "device_requested": device_str,
                "device_used": provider,
                "fallback_used": False,
                "fallback_reason": None,
                "engine_selection": selection,
                "warnings": warnings,
                "onnx": resolved,
            }
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            warnings.append("ONNX unavailable; used PyTorch fallback")
            resolved = {**resolved, "runtime_error": fallback_reason}
            selection = {
                **selection,
                "source": "fallback",
                "reason": "ONNX unavailable; PyTorch selected",
            }
            if requested == "onnxruntime" or selected_engine == "onnxruntime":
                depth = _infer_torch(img_bgr, model_id, device_str)
                return depth, {
                    "model_id": model_id,
                    "model_display_name": spec.display_name,
                    "engine_requested": requested,
                    "engine_used": "pytorch",
                    "device_requested": device_str,
                    "device_used": device_str,
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                    "engine_selection": selection,
                    "onnx_path": resolved.get("onnx_path") or resolved.get("expected_path"),
                    "warnings": warnings,
                    "onnx": resolved,
                }

    if requested == "onnxruntime" and selected_engine != "onnxruntime":
        warnings.append("ONNX unavailable; used PyTorch fallback")
        fallback_reason = selection.get("reason") or "ONNX unavailable"

    depth = _infer_torch(img_bgr, model_id, device_str)
    return depth, {
        "model_id": model_id,
        "model_display_name": spec.display_name,
        "engine_requested": requested,
        "engine_used": "pytorch",
        "device_requested": device_str,
        "device_used": device_str,
        "fallback_used": requested == "onnxruntime" and bool(warnings),
        "fallback_reason": fallback_reason,
        "engine_selection": selection,
        "onnx_path": resolved.get("onnx_path") or resolved.get("expected_path"),
        "warnings": warnings,
        "onnx": resolved,
    }


def _infer(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Compatibility helper returning only normalized depth."""

    depth, _metadata = _infer_with_metadata(img_bgr, model_name, device_str)
    return depth


def infer_depth_arrays(
    raw: bytes,
    model: str,
    device: str,
    filename: str | None = None,
    max_dim: int | None = None,
    engine_requested: str = "auto",
) -> dict[str, Any]:
    """Decode an image and return reusable normalized depth arrays via the depth cache."""

    model_id = normalize_model_id(model)
    spec = get_model_spec(model_id)
    engine_requested = normalize_engine_mode(engine_requested)
    depth_key = _depth_cache_key(raw, model_id, device, max_dim, engine_requested)
    with observability.trace_span(
        "inference",
        "depth_cache_lookup",
        {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
    ):
        cached_depth = _get_cached_depth(depth_key)
    with observability.trace_span(
        "inference",
        "decode",
        {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
    ):
        img = _decode(raw, max_dim=max_dim)
    if cached_depth is None:
        t0 = time.perf_counter()
        with observability.trace_span(
            "inference",
            "model_inference",
            {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
        ):
            if (engine_requested or "auto").lower() == "auto":
                depth, engine_metadata = _infer_with_metadata(img, model_id, device)
            else:
                depth, engine_metadata = _infer_with_metadata(
                    img, model_id, device, engine_requested
                )
        lat = round((time.perf_counter() - t0) * 1000, 1)
        resolution = {"width": img.shape[1], "height": img.shape[0]}
        _set_cached_depth(depth_key, depth, resolution)
        depth_cached = False
    else:
        depth, resolution = cached_depth
        lat = 0.0
        depth_cached = True
        engine_metadata = {
            "model_id": model_id,
            "model_display_name": spec.display_name,
            "engine_requested": engine_requested,
            "engine_used": "cache",
            "device_requested": device,
            "device_used": device,
            "fallback_used": False,
            "fallback_reason": None,
            "engine_selection": {
                "selected_engine": "cache",
                "source": "cache",
                "reason": "Depth cache hit",
            },
            "warnings": [],
        }

    return {
        "img_bgr": img,
        "depth": depth.astype(np.float32, copy=False),
        "latency_ms": lat,
        "model": model_id,
        "model_id": model_id,
        "model_display_name": spec.display_name,
        "device_used": engine_metadata.get("device_used", device),
        "resolution": resolution,
        "filename": filename,
        "depth_cached": depth_cached,
        **engine_metadata,
    }


def process_image(
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
    engine_requested: str = "auto",
) -> dict[str, Any]:
    """Decode, infer, colorize, and package one image response."""

    model_label = model
    try:
        arrays = infer_depth_arrays(
            raw,
            model,
            device,
            filename=filename,
            max_dim=max_dim,
            engine_requested=engine_requested,
        )
        model_id = arrays["model_id"]
        model_label = model_id
        spec = get_model_spec(model_id)
        metrics_mode = normalize_metrics_mode(metrics)
        output_set = parse_outputs(outputs)
        img = arrays["img_bgr"]
        depth = arrays["depth"]
        lat = arrays["latency_ms"]
        resolution = arrays["resolution"]
        depth_cached = arrays["depth_cached"]
        engine_metadata = {
            key: value
            for key, value in arrays.items()
            if key
            not in {
                "img_bgr",
                "depth",
                "latency_ms",
                "model",
                "resolution",
                "filename",
                "depth_cached",
            }
        }

        gt_result: dict[str, Any] | None = None
        gt_metadata: dict[str, Any] = {"provided": False}
        gt_visualizations: dict[str, str] = {}
        if gt_raw:
            with observability.trace_span(
                "inference",
                "gt_metrics",
                {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
            ):
                try:
                    gt_payload = decode_ground_truth(
                        gt_raw,
                        gt_filename,
                        invalid_value=gt_invalid_value,
                        scale=gt_scale,
                    )
                    gt_result = compute_ground_truth_metrics(
                        depth,
                        gt_payload.depth,
                        metadata=gt_payload.metadata,
                        invalid_value=gt_invalid_value,
                    )
                    gt_metadata = gt_result["metadata"]
                    gt_visualizations = gt_result.get("visualizations", {})
                except GroundTruthError:
                    raise
                except Exception as exc:
                    raise GroundTruthError(
                        f"Ground-truth metric computation failed: {exc}"
                    ) from exc
        elif gt_required:
            raise GroundTruthError("Ground-truth mode requires a GT depth file")

        with observability.trace_span(
            "inference",
            "metrics_computation",
            {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
        ):
            metric_payload = _metrics_for_mode(depth, img, metrics_mode, gt_result=gt_result)
        payload: dict[str, Any] = {
            "metrics": metric_payload,
            "latency_ms": lat,
            "model": model_id,
            "model_id": model_id,
            "model_display_name": spec.display_name,
            "colormap": colormap,
            "device_used": device,
            "resolution": resolution,
            "filename": filename,
            "cached": False,
            "depth_cached": depth_cached,
            "metrics_mode": metrics_mode,
            "outputs": list(output_set),
            "gt_metadata": gt_metadata,
            **engine_metadata,
        }
        payload.update(gt_visualizations)
        if "color" in output_set:
            with observability.trace_span(
                "inference",
                "color_encoding",
                {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
            ):
                payload["depth_map"] = _b64(_colorize(depth, colormap))
        if "gray" in output_set:
            with observability.trace_span(
                "inference",
                "grayscale_encoding",
                {"model_id": model_id, "device_type": observability.normalize_device_type(device)},
            ):
                gray = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                payload["grayscale"] = _b64(gray)
        observability.safe_observe(
            "service_inference",
            observability.record_inference,
            model_id,
            str(engine_metadata.get("engine_used", "pytorch")),
            str(engine_metadata.get("device_used", device)),
            lat,
            pixels=int(resolution.get("width", 0)) * int(resolution.get("height", 0)),
            cached=bool(depth_cached),
            outcome="ok",
            metrics_mode=metrics_mode,
            outputs_count=len(output_set),
            warnings_count=len(engine_metadata.get("warnings") or []),
        )
        return payload
    except Exception as exc:
        observability.safe_observe(
            "service_inference_crash",
            observability.record_crash,
            "inference",
            "INFERENCE_FAILED",
            exc,
        )
        observability.safe_observe(
            "service_inference_error",
            observability.record_inference,
            model_label,
            "unknown",
            device,
            None,
            outcome="error",
            error_code="INFERENCE_FAILED",
        )
        raise


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
    engine_requested: str = "auto",
) -> dict[str, Any]:
    """Run the blocking decode/inference/encode pipeline off the ASGI event loop."""

    async with _INFERENCE_SEMAPHORE:
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
                engine_requested,
            ),
        )
