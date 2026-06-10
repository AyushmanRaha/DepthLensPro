"""Ground-truth depth decoding, alignment, metrics, and visualizations."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, NamedTuple, cast

import cv2
import numpy as np
from PIL import Image

GT_MAX_SIZE_MB = 20
GT_SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".npy"}
GT_EPS = 1e-6


class GroundTruthError(ValueError):
    """Raised when an uploaded ground-truth file cannot be validated."""


class GroundTruthPayload(NamedTuple):
    depth: np.ndarray
    metadata: dict[str, Any]


def _b64_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise GroundTruthError("Could not encode ground-truth visualization")
    return base64.b64encode(buf.tobytes()).decode()


def _normalize_plane(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    if mask is None:
        mask = np.isfinite(arr)
    valid = arr[mask]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(np.nanmin(valid))
    hi = float(np.nanmax(valid))
    if hi - lo <= GT_EPS:
        return np.zeros(arr.shape, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return cast(np.ndarray, np.clip(out, 0.0, 1.0).astype(np.float32))


def _colorize(values: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> str:
    norm = _normalize_plane(values)
    u8 = (norm * 255).astype(np.uint8)
    return _b64_png(cv2.applyColorMap(u8, colormap))


def _decode_image_depth(raw: bytes, ext: str, warnings: list[str]) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as image:
        arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] in {3, 4}:
            warnings.append(
                "Ground-truth file was multi-channel; converted to grayscale for depth metrics."
            )
            # Preserve numeric range through luminance conversion instead of PIL mode conversion.
            arrf = arr[:, :, :3].astype(np.float32)
            arr = 0.299 * arrf[:, :, 0] + 0.587 * arrf[:, :, 1] + 0.114 * arrf[:, :, 2]
        else:
            raise GroundTruthError(f"Unsupported {ext} ground-truth channel count: {arr.shape}")
    if arr.ndim != 2:
        raise GroundTruthError(f"Ground-truth depth must be 2D; got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _decode_npy_depth(raw: bytes) -> np.ndarray:
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # pragma: no cover - exact numpy parser exception varies
        raise GroundTruthError(f"Could not decode NPY ground-truth depth: {exc}") from exc
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise GroundTruthError(f"NPY ground-truth depth must be H×W or H×W×1; got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise GroundTruthError(f"NPY ground-truth depth must be numeric; got {arr.dtype}")
    return cast(np.ndarray, arr.astype(np.float32, copy=False))


def decode_ground_truth(
    raw: bytes,
    filename: str | None,
    *,
    invalid_value: float | None = None,
    scale: float | None = None,
) -> GroundTruthPayload:
    """Decode PNG/TIFF/NPY ground truth into a finite 2D float32 depth plane."""

    if len(raw) / 1024**2 > GT_MAX_SIZE_MB:
        raise GroundTruthError(f"Ground-truth file exceeds {GT_MAX_SIZE_MB} MB limit")
    ext = Path(filename or "").suffix.lower()
    if ext not in GT_SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(GT_SUPPORTED_EXTENSIONS))
        raise GroundTruthError(
            f"Unsupported ground-truth format {ext or '(none)'}. Supported: {supported}"
        )

    warnings: list[str] = []
    depth = _decode_npy_depth(raw) if ext == ".npy" else _decode_image_depth(raw, ext, warnings)
    original_shape = [int(depth.shape[0]), int(depth.shape[1])]
    if scale and scale > 0:
        depth = depth * float(scale)
        warnings.append(f"Applied GT scale factor {scale:g}.")

    finite = np.isfinite(depth)
    if invalid_value is not None:
        finite &= depth != float(invalid_value)
    valid = finite & (depth > GT_EPS)
    if not bool(valid.any()):
        raise GroundTruthError("Ground-truth depth has no finite positive valid pixels")

    invalid_count = int(depth.size - int(valid.sum()))
    metadata = {
        "provided": True,
        "filename": filename,
        "format": ext.lstrip("."),
        "original_shape": original_shape,
        "decoded_shape": original_shape,
        "dtype": str(depth.dtype),
        "valid_pixel_count": int(valid.sum()),
        "invalid_pixel_count": invalid_count,
        "warnings": warnings,
    }
    return GroundTruthPayload(depth.astype(np.float32, copy=False), metadata)


def _valid_mask(gt: np.ndarray, invalid_value: float | None = None) -> np.ndarray:
    mask = np.isfinite(gt) & (gt > GT_EPS)
    if invalid_value is not None:
        mask &= gt != float(invalid_value)
    return cast(np.ndarray, mask)


def _align_gt_to_prediction(
    gt: np.ndarray, pred_shape: tuple[int, int], warnings: list[str]
) -> tuple[np.ndarray, str]:
    """Resize GT to prediction H×W with nearest-neighbor sampling.

    This keeps the model output grid stable for image responses and avoids inventing
    interpolated benchmark labels. The policy is documented in README.
    """

    if gt.shape == pred_shape:
        return gt.astype(np.float32, copy=False), "gt_resize_to_prediction:none"
    resized = cv2.resize(
        gt.astype(np.float32, copy=False),
        (pred_shape[1], pred_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    warnings.append(
        "Resized GT from "
        f"{gt.shape[1]}×{gt.shape[0]} to {pred_shape[1]}×{pred_shape[0]} "
        "using nearest-neighbor alignment."
    )
    return resized.astype(np.float32, copy=False), "gt_resize_to_prediction:nearest"


def compute_ground_truth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    metadata: dict[str, Any],
    invalid_value: float | None = None,
) -> dict[str, Any]:
    """Align relative prediction to GT with median scale and compute benchmark metrics."""

    warnings = list(metadata.get("warnings") or [])
    gt_aligned, align_policy = _align_gt_to_prediction(gt, pred.shape[:2], warnings)
    valid = _valid_mask(gt_aligned, invalid_value)
    pred32 = pred.astype(np.float32, copy=False)
    valid &= np.isfinite(pred32) & (pred32 > GT_EPS)
    if not bool(valid.any()):
        raise GroundTruthError("No overlapping finite positive prediction/GT pixels for metrics")

    pred_valid = pred32[valid].astype(np.float64)
    gt_valid = gt_aligned[valid].astype(np.float64)
    pred_median = float(np.median(pred_valid[pred_valid > GT_EPS]))
    gt_median = float(np.median(gt_valid[gt_valid > GT_EPS]))
    if pred_median <= GT_EPS or gt_median <= GT_EPS:
        raise GroundTruthError("Could not compute median scale alignment for GT metrics")
    scale = gt_median / pred_median
    pred_scaled = pred32.astype(np.float64) * scale
    pv = pred_scaled[valid]
    gv = gt_valid
    diff = pv - gv
    abs_diff = np.abs(diff)
    sq_diff = diff**2
    ratio = np.maximum(pv / (gv + GT_EPS), gv / (pv + GT_EPS))

    metrics = {
        "abs_rel": round(float(np.mean(abs_diff / (gv + GT_EPS))), 4),
        "sq_rel": round(float(np.mean(sq_diff / (gv + GT_EPS))), 4),
        "gt_mae": round(float(np.mean(abs_diff)), 4),
        "gt_rmse": round(float(np.sqrt(np.mean(sq_diff))), 4),
        "gt_log_rmse": round(
            float(np.sqrt(np.mean((np.log(pv + GT_EPS) - np.log(gv + GT_EPS)) ** 2))), 4
        ),
        "delta_1": round(float(np.mean(ratio < 1.25)), 4),
        "delta_2": round(float(np.mean(ratio < 1.25**2)), 4),
        "delta_3": round(float(np.mean(ratio < 1.25**3)), 4),
    }

    err = np.zeros_like(pred32, dtype=np.float32)
    err[valid] = np.abs(pred_scaled.astype(np.float32)[valid] - gt_aligned[valid])
    gt_vis = _colorize(gt_aligned)
    err_vis = _colorize(err, cv2.COLORMAP_TURBO)

    updated_meta = {
        **metadata,
        "aligned_shape": [int(gt_aligned.shape[0]), int(gt_aligned.shape[1])],
        "valid_pixel_count": int(valid.sum()),
        "invalid_pixel_count": int(valid.size - int(valid.sum())),
        "alignment_policy": align_policy,
        "scale_alignment": "median_scale",
        "scale_factor": round(float(scale), 6),
        "warnings": warnings,
    }
    unavailable = {
        "gt_ssim": "not_implemented",
        "gt_psnr": "not_implemented",
        "ordinal_error": "not_implemented",
        "surface_normal_error": "not_implemented",
        "lpips": "not_implemented",
    }
    return {
        "metrics": metrics,
        "metadata": updated_meta,
        "visualizations": {"gt_depth_map": gt_vis, "error_heatmap": err_vis},
        "unavailable": unavailable,
        "warnings": warnings,
    }
