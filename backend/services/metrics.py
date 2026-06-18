"""Depth metric parsing, computation, and grouping helpers."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from backend.config import settings
from backend.constants import SUPPORTED_METRICS_MODES, SUPPORTED_OUTPUT_MODES


def normalize_metrics_mode(mode: str | None) -> str:
    value = (mode or settings.DEPTHLENS_DEFAULT_METRICS or "fast").strip().lower()
    if value not in SUPPORTED_METRICS_MODES:
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
        if item not in SUPPORTED_OUTPUT_MODES:
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

    hist256, edges256 = np.histogram(flat, bins=256, range=(0.0, 1.0))
    hist_total = max(float(hist256.sum()), 1.0)
    hp = hist256 / hist_total
    nonzero_hp = hp[hp > 0]
    entropy = float(-np.sum(nonzero_hp * np.log2(nonzero_hp)))
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

    # Reuse the 256-bin histogram for the public 32-bin payload instead of
    # scanning the depth plane a second time.  The ranges align exactly.
    h32 = hist256.reshape(32, 8).sum(axis=1)
    edges = edges256[::8]

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
