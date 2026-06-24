"""Request validation helpers for DepthLens Pro API routes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePath

from fastapi import HTTPException, UploadFile

from backend.api.errors import http_error
from backend.config import settings
from backend.constants import MAX_UPLOAD_SIZE_MB
from backend.model_metadata import COLORMAP_NAMES
from backend.model_registry import UnknownModelError, normalize_model_id

GT_EXTENSIONS = {".png", ".tif", ".tiff", ".npy"}


def validate_image_upload_content_type(file: UploadFile) -> None:
    if not (file.content_type or "").startswith("image/"):
        raise http_error(415, "INVALID_CONTENT_TYPE", "Expected an image file", field="file")


def validate_upload_size(
    raw: bytes, *, label: str = "Image file", max_size_mb: int = MAX_UPLOAD_SIZE_MB
) -> None:
    if len(raw) > max_size_mb * 1024 * 1024:
        raise http_error(
            413, "UPLOAD_TOO_LARGE", f"{label} exceeds {max_size_mb} MB limit", field="file"
        )


def validate_gt_upload(file: UploadFile, raw: bytes) -> None:
    suffix = PurePath(file.filename or "").suffix.lower()
    if suffix not in GT_EXTENSIONS:
        raise http_error(
            422, "INVALID_GT_FILE", "GT depth file must be PNG, TIFF, or NPY", field="gt_file"
        )
    validate_upload_size(raw, label="GT depth file", max_size_mb=MAX_UPLOAD_SIZE_MB)


def validate_max_dim(max_dim: int | None) -> None:
    if max_dim is not None and not (64 <= max_dim <= settings.DEPTHLENS_MAX_DIM):
        raise http_error(
            422,
            "INVALID_MAX_DIM",
            f"max_dim must be between 64 and {settings.DEPTHLENS_MAX_DIM}",
            field="max_dim",
        )


def validate_detection_params(threshold: float, max_detections: int) -> None:
    if threshold < 0.05 or threshold > 0.95:
        raise http_error(
            422,
            "INVALID_DETECTION_THRESHOLD",
            "threshold must be between 0.05 and 0.95",
            field="threshold",
        )
    if max_detections < 1 or max_detections > 20:
        raise http_error(
            422,
            "INVALID_MAX_DETECTIONS",
            "max_detections must be between 1 and 20",
            field="max_detections",
        )


def validate_reconstruction_params(
    max_points: int,
    preview_points: int,
    focal_scale: float,
    depth_scale: float,
    near: float,
    far: float,
    sampling: str,
    coordinate_system: str,
) -> None:
    if max_points < 1000 or max_points > 2_000_000:
        raise http_error(
            422,
            "INVALID_RECONSTRUCTION_PARAMETER",
            "max_points must be between 1000 and 2000000",
            field="max_points",
        )
    if preview_points < 100 or preview_points > max_points:
        raise http_error(
            422,
            "INVALID_RECONSTRUCTION_PARAMETER",
            "preview_points must be between 100 and max_points",
            field="preview_points",
        )
    if focal_scale <= 0 or depth_scale <= 0:
        raise http_error(
            422,
            "INVALID_RECONSTRUCTION_PARAMETER",
            "focal_scale and depth_scale must be positive",
            field="focal_scale",
        )
    if not (0 <= near < far <= 100):
        raise http_error(
            422,
            "INVALID_RECONSTRUCTION_PARAMETER",
            "depth percentiles must satisfy 0 <= near < far <= 100",
            field="depth_near_percentile",
        )
    if sampling not in {"grid", "stride", "random"}:
        raise http_error(
            422,
            "INVALID_RECONSTRUCTION_PARAMETER",
            "sampling must be grid, stride, or random",
            field="sampling",
        )
    if coordinate_system not in {"y_up", "z_up"}:
        raise http_error(
            422,
            "INVALID_RECONSTRUCTION_PARAMETER",
            "coordinate_system must be y_up or z_up",
            field="coordinate_system",
        )


def normalize_request_model(model: str) -> str:
    try:
        return normalize_model_id(model)
    except UnknownModelError as exc:
        raise HTTPException(
            422,
            {"error_code": exc.error_code, "message": str(exc), "valid_models": exc.valid_models},
        ) from exc


def normalize_request_colormap(colormap: str) -> str:
    if colormap not in COLORMAP_NAMES:
        raise http_error(
            422,
            "INVALID_COLORMAP",
            f"Unknown colormap '{colormap}'",
            field="colormap",
            valid_colormaps=list(COLORMAP_NAMES),
        )
    return colormap


def normalize_request_metrics_and_outputs(
    metrics: str | None,
    outputs: str | None,
    *,
    normalize_metrics_mode: Callable[[str], str],
    parse_outputs: Callable[[str], list[str]],
) -> tuple[str, str]:
    metrics_value = settings.DEPTHLENS_DEFAULT_METRICS if metrics is None else metrics
    outputs_value = settings.DEPTHLENS_DEFAULT_OUTPUTS if outputs is None else outputs
    return str(normalize_metrics_mode(metrics_value)), ",".join(parse_outputs(outputs_value))
