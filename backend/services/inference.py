"""Depth-estimation inference and image post-processing services."""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import time
from typing import Any, Callable, cast

import cv2
import numpy as np
import torch

from backend.depth_models import ONNXExecutionEngine, onnx_model_path

log = logging.getLogger("depthlens")

SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "MiDaS_small": {
        "label": "Small",
        "description": "~30 ms · EfficientNet-Lite · CPU-friendly",
        "compute": "CPU or GPU",
    },
    "DPT_Hybrid": {
        "label": "Hybrid",
        "description": "~120 ms · ViT-Hybrid · GPU recommended",
        "compute": "GPU recommended",
    },
    "DPT_Large": {
        "label": "Large",
        "description": "~400 ms · ViT-Large · GPU required for speed",
        "compute": "GPU required",
    },
}

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

MAX_DIM = 2048
MAX_SIZE_MB = 20
MODELS: dict[str, tuple[torch.nn.Module, torch.device]] = {}
ONNX_ENGINES: dict[str, ONNXExecutionEngine] = {}
TRANSFORMS: dict[str, Callable[[np.ndarray], torch.Tensor]] = {}


def clear_models() -> None:
    """Release all loaded model and transform references."""
    MODELS.clear()
    ONNX_ENGINES.clear()
    TRANSFORMS.clear()


def loaded_model_keys() -> list[str]:
    """Return cache keys for models currently loaded in memory."""
    return list(MODELS.keys()) + [f"onnx:{key}" for key in ONNX_ENGINES]


def _load_model(
    model_name: str, device_str: str
) -> tuple[tuple[torch.nn.Module, torch.device], Callable[[np.ndarray], torch.Tensor]]:
    """Load or reuse a MiDaS model for a resolved device."""
    key = f"{model_name}:{device_str}"
    if key in MODELS:
        return MODELS[key], TRANSFORMS[model_name]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")

    device = torch.device(device_str)
    log.info("Loading '%s' → %s …", model_name, device)
    model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)  # type: ignore[no-untyped-call]
    model.to(device).eval()

    if device.type == "mps":
        model = model.float()

    if model_name not in TRANSFORMS:
        transforms = torch.hub.load(  # type: ignore[no-untyped-call]
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        TRANSFORMS[model_name] = (
            transforms.small_transform if model_name == "MiDaS_small" else transforms.dpt_transform
        )

    MODELS[key] = (model, device)
    log.info("✅ '%s' ready on %s", model_name, device)
    return (model, device), TRANSFORMS[model_name]


def _decode(raw: bytes) -> np.ndarray:
    """Decode image bytes into a BGR OpenCV image, resizing oversized inputs."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — corrupt or unsupported format")
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a raw depth plane into the existing [0, 1] output range."""

    depth = depth.astype(np.float32, copy=False)
    lo, hi = depth.min(), depth.max()
    return cast(np.ndarray, (depth - lo) / (hi - lo + 1e-8))


def _infer_torch(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Run normalized PyTorch depth inference for benchmarking and fallback."""

    (model, device), transform = _load_model(model_name, device_str)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)
    with torch.no_grad():
        pred = model(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return _normalize_depth(pred.cpu().numpy().astype(np.float32))


def _load_onnx_engine(model_name: str, device_str: str) -> ONNXExecutionEngine:
    """Load or reuse an ONNX Runtime execution engine for static MiDaS weights."""

    key = f"{model_name}:{device_str}"
    if key not in ONNX_ENGINES:
        ONNX_ENGINES[key] = ONNXExecutionEngine(model_name=model_name, device=device_str)
    return ONNX_ENGINES[key]


def _infer_onnx(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Run normalized ONNX Runtime depth inference for a BGR image."""

    engine = _load_onnx_engine(model_name, device_str)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return _normalize_depth(engine.forward(rgb))


def _infer(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Run normalized depth inference, preferring ONNX Runtime static weights."""

    if onnx_model_path(model_name).exists():
        return _infer_onnx(img_bgr, model_name, device_str)
    log.warning("ONNX weights unavailable for %s; falling back to PyTorch execution", model_name)
    return _infer_torch(img_bgr, model_name, device_str)


def _colorize(depth: np.ndarray, cmap: str) -> np.ndarray:
    """Apply an OpenCV color map to a normalized depth map."""
    u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, COLORMAPS.get(cmap, cv2.COLORMAP_INFERNO))


def _b64(img: np.ndarray) -> str:
    """Encode an image as a PNG base64 string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _fhash(raw: bytes, model: str, cmap: str, dev: str) -> str:
    """Build a stable cache key for request image bytes and inference options."""
    return hashlib.sha1(f"{model}:{cmap}:{dev}:{hashlib.md5(raw).hexdigest()}".encode()).hexdigest()


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
    raw: bytes, model: str, colormap: str, device: str, filename: str | None
) -> dict[str, Any]:
    """Decode, infer, colorize, and package one image response."""
    img = _decode(raw)
    t0 = time.perf_counter()
    depth = _infer(img, model, device)
    lat = round((time.perf_counter() - t0) * 1000, 1)
    col = _colorize(depth, colormap)
    gray = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return {
        "depth_map": _b64(col),
        "grayscale": _b64(gray),
        "metrics": _compute_metrics(depth, img),
        "latency_ms": lat,
        "model": model,
        "colormap": colormap,
        "device_used": device,
        "resolution": {"width": img.shape[1], "height": img.shape[0]},
        "filename": filename,
        "cached": False,
    }
