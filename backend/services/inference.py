"""Depth-estimation inference and image post-processing services."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import os
import threading
import time
from typing import Any, Callable, cast

import cv2
import numpy as np
import torch
from starlette.concurrency import run_in_threadpool

from backend.config import settings
from backend.depth_models import ONNXExecutionEngine
from backend.model_metadata import COLORMAP_NAMES, SUPPORTED_MODELS
from backend.model_registry import get_model_spec, normalize_model_id, resolve_onnx_path
from backend.services.ground_truth import (
    GroundTruthError,
    compute_ground_truth_metrics,
    decode_ground_truth,
)

log = logging.getLogger("depthlens")

__all__ = ["COLORMAPS", "MAX_SIZE_MB", "SUPPORTED_MODELS"]

COLORMAPS: dict[str, int] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    "turbo": cv2.COLORMAP_TURBO,
}
assert set(COLORMAPS) == set(COLORMAP_NAMES)

MAX_DIM = int(settings.DEPTHLENS_MAX_DIM)
MAX_SIZE_MB = 20
MODELS: dict[str, tuple[torch.nn.Module, torch.device]] = {}
ONNX_ENGINES: dict[str, ONNXExecutionEngine] = {}
_DEPTH_CACHE: dict[str, tuple[float, np.ndarray, dict[str, int]]] = {}
_DEPTH_CACHE_MAX_ENTRIES = 12
_ONNX_MISSING_WARNED: set[str] = set()
TRANSFORMS: dict[str, Callable[[np.ndarray], torch.Tensor]] = {}
_MODEL_LOCK = threading.RLock()
_MODEL_FORWARD_LOCKS: dict[str, threading.Lock] = {}
_ONNX_LOCK = threading.RLock()
_ONNX_FORWARD_LOCKS: dict[str, threading.Lock] = {}
_INFERENCE_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("INFERENCE_MAX_CONCURRENCY", "2"))))


def clear_models() -> None:
    """Release all loaded model and transform references."""
    with _MODEL_LOCK, _ONNX_LOCK:
        MODELS.clear()
        ONNX_ENGINES.clear()
        TRANSFORMS.clear()
        _MODEL_FORWARD_LOCKS.clear()
        _ONNX_FORWARD_LOCKS.clear()
        _DEPTH_CACHE.clear()
        _ONNX_MISSING_WARNED.clear()


def loaded_model_keys() -> list[str]:
    """Return cache keys for models currently loaded in memory."""
    return list(MODELS.keys()) + [f"onnx:{key}" for key in ONNX_ENGINES]


def _load_model(
    model_name: str, device_str: str
) -> tuple[tuple[torch.nn.Module, torch.device], Callable[[np.ndarray], torch.Tensor]]:
    """Load or reuse a MiDaS model for a resolved device."""
    model_id = normalize_model_id(model_name)
    spec = get_model_spec(model_id)
    key = f"{model_id}:{device_str}"
    with _MODEL_LOCK:
        if key in MODELS:
            return MODELS[key], TRANSFORMS[model_id]

        # MiDaS preprocessing transforms are selected by model family only and do
        # not hold device state.  Keep the cache model-scoped intentionally; if a
        # future transform depends on CUDA/MPS/XPU state, change this key to
        # include ``device_str`` alongside the model cache key below.
        if model_id not in TRANSFORMS:
            transforms = torch.hub.load(  # type: ignore[no-untyped-call]
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            TRANSFORMS[model_id] = (
                transforms.small_transform
                if model_id == "midas_small"
                else transforms.dpt_transform
            )

        device = torch.device(device_str)
        log.info("Loading '%s' (%s) → %s …", spec.display_name, spec.pytorch_model_name, device)
        model = torch.hub.load("intel-isl/MiDaS", spec.pytorch_model_name, trust_repo=True)  # type: ignore[no-untyped-call]
        model.to(device).eval()

        if device.type == "mps":
            model = model.float()

        MODELS[key] = (model, device)
        _MODEL_FORWARD_LOCKS.setdefault(key, threading.Lock())
        log.info("✅ '%s' ready on %s", spec.display_name, device)
        return (model, device), TRANSFORMS[model_id]


def _decode(raw: bytes, max_dim: int | None = None) -> np.ndarray:
    """Decode image bytes into a BGR OpenCV image, resizing oversized inputs."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — corrupt or unsupported format")
    h, w = img.shape[:2]
    limit = max(256, int(max_dim or MAX_DIM))
    if max(h, w) > limit:
        scale = limit / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a raw depth plane into the existing [0, 1] output range."""

    depth = depth.astype(np.float32, copy=False)
    lo, hi = depth.min(), depth.max()
    return cast(np.ndarray, (depth - lo) / (hi - lo + 1e-8))


def _infer_torch(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Run normalized PyTorch depth inference for benchmarking and fallback."""

    model_id = normalize_model_id(model_name)
    (model, device), transform = _load_model(model_id, device_str)
    key = f"{model_id}:{device_str}"
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device, non_blocking=True)
    with _MODEL_FORWARD_LOCKS[key], torch.inference_mode():
        pred = model(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    del batch, pred
    return _normalize_depth(depth)


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
            fallback_reason = str(resolved.get("error") or "missing_file")
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


def _infer(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Compatibility helper returning only normalized depth."""

    depth, _metadata = _infer_with_metadata(img_bgr, model_name, device_str)
    return depth


def _colorize(depth: np.ndarray, cmap: str) -> np.ndarray:
    """Apply an OpenCV color map to a normalized depth map."""
    u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, COLORMAPS.get(cmap, cv2.COLORMAP_INFERNO))


def _b64(img: np.ndarray) -> str:
    """Encode an image as a PNG base64 string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _raw_hash(raw: bytes) -> str:
    """Return a stable content hash for raw image bytes."""
    return hashlib.sha256(raw).hexdigest()


def _depth_cache_key(raw: bytes, model: str, dev: str, max_dim: int | None) -> str:
    """Cache normalized depth independently from color/output/metric options."""
    limit = max(256, int(max_dim or MAX_DIM))
    return hashlib.sha1(f"depth:{model}:{dev}:{limit}:{_raw_hash(raw)}".encode()).hexdigest()


def _fhash(
    raw: bytes,
    model: str,
    cmap: str,
    dev: str,
    metrics: str = "full",
    outputs: str = "color,gray",
    max_dim: int | None = None,
) -> str:
    """Build a stable cache key for request image bytes and response options."""
    limit = max(256, int(max_dim or MAX_DIM))
    output_key = ",".join(parse_outputs(outputs))
    metric_key = normalize_metrics_mode(metrics)
    return hashlib.sha1(
        f"{model}:{cmap}:{dev}:{metric_key}:{output_key}:{limit}:{_raw_hash(raw)}".encode()
    ).hexdigest()


def normalize_metrics_mode(mode: str | None) -> str:
    value = (mode or settings.DEPTHLENS_DEFAULT_METRICS or "fast").strip().lower()
    if value not in {"none", "fast", "full"}:
        raise ValueError("metrics must be one of: none, fast, full")
    return value


def parse_outputs(outputs: str | None) -> tuple[str, ...]:
    raw = outputs if outputs is not None else settings.DEPTHLENS_DEFAULT_OUTPUTS
    requested = [part.strip().lower() for part in str(raw or "color").split(",") if part.strip()]
    if not requested:
        requested = ["color"]
    normalized: list[str] = []
    for item in requested:
        if item in {"depth", "depth_map"}:
            item = "color"
        if item in {"grayscale", "grey"}:
            item = "gray"
        if item not in {"color", "gray"}:
            raise ValueError("outputs must contain only color and/or gray")
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01) ** 2, (0.03) ** 2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x.astype(np.float32), (11, 11), 1.5).astype(np.float64)

    mu_a, mu_b = blur(a), blur(b)
    sig_a = blur(a**2) - mu_a**2
    sig_b = blur(b**2) - mu_b**2
    sig_ab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(np.mean(num / (den + 1e-10)))


GT_METRIC_KEYS = {
    "abs_rel",
    "sq_rel",
    "gt_mae",
    "gt_rmse",
    "gt_log_rmse",
    "delta_1",
    "delta_2",
    "delta_3",
    "gt_ssim",
    "gt_psnr",
    "lpips",
    "ordinal_error",
    "surface_normal_error",
}
PROXY_METRIC_KEYS = {
    "mae",
    "rmse",
    "log_rmse",
    "silog",
    "psnr",
    "ssim",
    "gradient_error",
}
FULL_ONLY_KEYS = {
    "median",
    "mae",
    "rmse",
    "log_rmse",
    "silog",
    "psnr",
    "dynamic_range",
    "coverage",
    "ssim",
    "gradient_mean",
    "gradient_std",
    "gradient_error",
    "edge_density",
}


def _with_metric_groups(
    flat: dict[str, Any],
    mode: str,
    *,
    gt_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction_keys = {
        "min",
        "max",
        "mean",
        "std",
        "median",
        "dynamic_range",
        "entropy",
        "coverage",
        "histogram",
    }
    proxy_keys = PROXY_METRIC_KEYS | {"gradient_mean", "gradient_std", "edge_density"}
    grouped: dict[str, Any] = {
        "prediction_stats": {k: flat[k] for k in prediction_keys if k in flat},
        "proxy_metrics": {k: flat[k] for k in proxy_keys if k in flat},
        "gt_metrics": {},
        "unavailable": {},
        "warnings": [],
    }
    if mode == "fast":
        for key in sorted(FULL_ONLY_KEYS):
            grouped["unavailable"][key] = "not_requested_fast_mode"
    for key in sorted(GT_METRIC_KEYS):
        grouped["unavailable"][key] = "needs_gt_depth_upload"
    if gt_result:
        flat.update(gt_result.get("metrics", {}))
        grouped["gt_metrics"].update(gt_result.get("metrics", {}))
        grouped["warnings"].extend(gt_result.get("warnings", []))
        for key, reason in gt_result.get("unavailable", {}).items():
            grouped["unavailable"][key] = reason
        for key in grouped["gt_metrics"]:
            grouped["unavailable"].pop(key, None)
    flat.update(grouped)
    return flat


def _compute_fast_metrics(depth: np.ndarray) -> dict[str, Any]:
    """Compute lightweight metrics for interactive workspace inference."""
    depth32 = depth.astype(np.float32, copy=False)
    flat = depth32.ravel()
    hist, edges = np.histogram(flat, bins=32, range=(0.0, 1.0))
    hp = hist / max(float(hist.sum()), 1.0)
    entropy = float(-np.sum(hp[hp > 0] * np.log2(hp[hp > 0])))
    return {
        "min": round(float(flat.min()), 4),
        "max": round(float(flat.max()), 4),
        "mean": round(float(flat.mean()), 4),
        "std": round(float(flat.std()), 4),
        "entropy": round(entropy, 3),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": [round(float(e), 3) for e in edges.tolist()],
        },
    }


def _metrics_for_mode(
    depth: np.ndarray,
    img_bgr: np.ndarray,
    mode: str,
    *,
    gt_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode == "none":
        base: dict[str, Any] = {}
    elif mode == "fast":
        base = _compute_fast_metrics(depth)
    else:
        base = _compute_metrics(depth, img_bgr)
    return _with_metric_groups(base, mode, gt_result=gt_result)


def _get_cached_depth(cache_key: str) -> tuple[np.ndarray, dict[str, int]] | None:
    now = time.time()
    item = _DEPTH_CACHE.get(cache_key)
    if item is None:
        return None
    expires_at, depth, resolution = item
    if expires_at <= now:
        _DEPTH_CACHE.pop(cache_key, None)
        return None
    return depth.copy(), dict(resolution)


def _set_cached_depth(cache_key: str, depth: np.ndarray, resolution: dict[str, int]) -> None:
    if len(_DEPTH_CACHE) >= _DEPTH_CACHE_MAX_ENTRIES:
        oldest = min(_DEPTH_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _DEPTH_CACHE.pop(oldest, None)
    _DEPTH_CACHE[cache_key] = (
        time.time() + int(settings.CACHE_TTL_SECONDS),
        depth.copy(),
        dict(resolution),
    )


def _compute_metrics(depth: np.ndarray, img_bgr: np.ndarray) -> dict[str, Any]:
    """Compute the existing metric payload for a normalized depth map."""
    depth64 = depth.astype(np.float64)
    flat = depth64.flatten()

    d_min, d_max = float(flat.min()), float(flat.max())
    d_mean = float(flat.mean())
    d_std = float(flat.std())
    d_med = float(np.median(flat))

    nz = flat[flat > 1e-6]
    dyn = float(np.log2(nz.max() / nz.min())) if len(nz) > 0 else 0.0

    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0))
    hp = hist / hist.sum()
    entropy = float(-np.sum(hp[hp > 0] * np.log2(hp[hp > 0])))
    coverage = float((hp >= hp.max() * 0.01).mean())

    depth32 = depth64.astype(np.float32)
    gx = cv2.Sobel(depth32, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth32, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(gmag.mean())
    grad_std = float(gmag.std())
    grad_error = grad_mean
    edge_thresh = gmag.mean() + gmag.std()
    edge_density = float((gmag > edge_thresh).mean())

    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    ssim_val = _ssim(gray_img, depth64)

    pseudo = np.full_like(depth64, d_mean)
    eps = 1e-6
    mae = float(np.abs(depth64 - pseudo).mean())
    rmse = float(np.sqrt(np.mean((depth64 - pseudo) ** 2)))
    log_rmse = float(np.sqrt(np.mean((np.log(depth64 + eps) - np.log(pseudo + eps)) ** 2)))

    log_d = np.log(depth64 + eps)
    ld_mean = log_d.mean()
    silog = float(np.sqrt(np.mean((log_d - ld_mean) ** 2)) * 100)

    mse_v = float(np.mean((depth64 - d_mean) ** 2))
    psnr = float(10 * math.log10(1.0 / (mse_v + 1e-10))) if mse_v > 1e-10 else 99.0

    h32, edges = np.histogram(flat, bins=32, range=(0.0, 1.0))

    return {
        "min": round(d_min, 4),
        "max": round(d_max, 4),
        "mean": round(d_mean, 4),
        "std": round(d_std, 4),
        "median": round(d_med, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "log_rmse": round(log_rmse, 4),
        "silog": round(silog, 2),
        "psnr": round(psnr, 2),
        "dynamic_range": round(dyn, 2),
        "entropy": round(entropy, 3),
        "coverage": round(coverage, 4),
        "ssim": round(ssim_val, 4),
        "gradient_mean": round(grad_mean, 4),
        "gradient_std": round(grad_std, 4),
        "gradient_error": round(grad_error, 4),
        "edge_density": round(edge_density, 4),
        "abs_rel": None,
        "sq_rel": None,
        "delta_1": None,
        "delta_2": None,
        "delta_3": None,
        "lpips": None,
        "ordinal_error": None,
        "surface_normal_error": None,
        "histogram": {"counts": h32.tolist(), "bin_edges": [round(e, 3) for e in edges.tolist()]},
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
) -> dict[str, Any]:
    """Decode, infer, colorize, and package one image response."""
    model_id = normalize_model_id(model)
    spec = get_model_spec(model_id)
    metrics_mode = normalize_metrics_mode(metrics)
    output_set = parse_outputs(outputs)
    depth_key = _depth_cache_key(raw, model_id, device, max_dim)
    cached_depth = _get_cached_depth(depth_key)
    img = _decode(raw, max_dim=max_dim)
    if cached_depth is None:
        t0 = time.perf_counter()
        depth, engine_metadata = _infer_with_metadata(img, model_id, device)
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
            "engine_requested": "auto",
            "engine_used": "cache",
            "device_requested": device,
            "device_used": device,
            "fallback_used": False,
            "warnings": [],
        }

    gt_result: dict[str, Any] | None = None
    gt_metadata: dict[str, Any] = {"provided": False}
    gt_visualizations: dict[str, str] = {}
    if gt_raw:
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
            raise GroundTruthError(f"Ground-truth metric computation failed: {exc}") from exc
    elif gt_required:
        raise GroundTruthError("Ground-truth mode requires a GT depth file")

    payload: dict[str, Any] = {
        "metrics": _metrics_for_mode(depth, img, metrics_mode, gt_result=gt_result),
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
        payload["depth_map"] = _b64(_colorize(depth, colormap))
    if "gray" in output_set:
        gray = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        payload["grayscale"] = _b64(gray)
    return payload


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
    """Run the blocking decode/inference/encode pipeline off the ASGI event loop."""

    async with _INFERENCE_SEMAPHORE:
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
