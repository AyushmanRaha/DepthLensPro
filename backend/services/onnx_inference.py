"""ONNX Runtime engine cache, locking, inference, and fallback metadata."""

from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

from backend.depth_models import ONNXExecutionEngine
from backend.model_registry import get_model_spec, normalize_model_id, resolve_onnx_path
from backend.services.image_io import _normalize_depth
from backend.services.model_runtime import _infer_torch

log = logging.getLogger("depthlens")

ONNX_ENGINES: dict[str, ONNXExecutionEngine] = {}
_ONNX_MISSING_WARNED: set[str] = set()
_ONNX_LOCK = threading.RLock()
_ONNX_FORWARD_LOCKS: dict[str, threading.Lock] = {}


def _load_onnx_engine_with_lock(
    model_name: str, device_str: str
) -> tuple[ONNXExecutionEngine, threading.Lock]:
    """Load/reuse an ONNX engine and its matching forward lock atomically."""

    model_id = normalize_model_id(model_name)
    key = f"{model_id}:{device_str}"
    with _ONNX_LOCK:
        forward_lock = _ONNX_FORWARD_LOCKS.setdefault(key, threading.Lock())
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

    model_id = normalize_model_id(model_name)
    engine, forward_lock = _load_onnx_engine_with_lock(model_id, device_str)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with forward_lock:
        depth = engine.forward(rgb)
    return _normalize_depth(depth)


def _infer_with_metadata(
    img_bgr: np.ndarray, model_name: str, device_str: str, engine_requested: str = "auto"
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run depth inference with guarded ONNX and PyTorch fallback metadata."""

    model_id = normalize_model_id(model_name)
    spec = get_model_spec(model_id)
    requested = (engine_requested or "auto").lower()
    warnings: list[str] = []
    onnx_detail: dict[str, Any] | None = None
    if requested in {"auto", "onnx", "onnxruntime"}:
        resolved = resolve_onnx_path(model_id)
        onnx_detail = resolved
        if resolved.get("exists") and int(resolved.get("size_bytes") or 0) > 0:
            try:
                depth = _infer_onnx(img_bgr, model_id, device_str)
                return depth, {
                    "model_id": model_id,
                    "model_display_name": spec.display_name,
                    "engine_requested": requested,
                    "engine_used": "onnxruntime",
                    "device_requested": device_str,
                    "device_used": getattr(
                        _load_onnx_engine(model_id, device_str), "provider", "onnxruntime"
                    ),
                    "fallback_used": False,
                    "warnings": warnings,
                    "onnx": resolved,
                }
            except Exception as exc:
                warnings.append(f"ONNX unavailable ({type(exc).__name__}); used PyTorch fallback")
                fallback_reason = f"{type(exc).__name__}: {exc}"
                onnx_detail = {**resolved, "runtime_error": fallback_reason}
                log.warning(
                    "ONNX inference unavailable for %s; falling back to PyTorch: %s", model_id, exc
                )
        else:
            warnings.append("ONNX model missing; used PyTorch fallback")
            if model_id not in _ONNX_MISSING_WARNED:
                log.warning(
                    "ONNX weights unavailable for %s at %s; falling back to PyTorch execution",
                    model_id,
                    resolved.get("onnx_path"),
                )
                _ONNX_MISSING_WARNED.add(model_id)
    depth = _infer_torch(img_bgr, model_id, device_str)
    return depth, {
        "model_id": model_id,
        "model_display_name": spec.display_name,
        "engine_requested": requested,
        "engine_used": "pytorch",
        "device_requested": device_str,
        "device_used": device_str,
        "fallback_used": requested in {"auto", "onnx", "onnxruntime"} and bool(warnings),
        "fallback_reason": warnings[-1] if warnings else None,
        "onnx_path": (onnx_detail or {}).get("onnx_path")
        or (onnx_detail or {}).get("expected_path"),
        "warnings": warnings,
        "onnx": onnx_detail,
    }
