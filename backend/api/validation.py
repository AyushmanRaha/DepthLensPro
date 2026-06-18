"""Request validation helpers for DepthLens Pro API routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException, UploadFile

from backend.config import settings
from backend.constants import MAX_UPLOAD_SIZE_MB
from backend.model_metadata import COLORMAP_NAMES
from backend.model_registry import UnknownModelError, normalize_model_id


def validate_image_upload_content_type(file: UploadFile) -> None:
    """Validate that an upload advertises an image content type."""

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")


def validate_upload_size(
    raw: bytes, *, label: str = "Image file", max_size_mb: int = MAX_UPLOAD_SIZE_MB
) -> None:
    """Validate upload size against the public API limit."""

    if len(raw) > max_size_mb * 1024 * 1024:
        raise HTTPException(413, f"{label} exceeds {max_size_mb} MB limit")


def normalize_request_model(model: str) -> str:
    """Normalize a user-supplied model id or raise the public 422 payload."""

    try:
        return normalize_model_id(model)
    except UnknownModelError as exc:
        raise HTTPException(
            422,
            {"error_code": exc.error_code, "message": str(exc), "valid_models": exc.valid_models},
        ) from exc


def normalize_request_colormap(colormap: str) -> str:
    """Validate a user-supplied colormap without changing existing messages."""

    if colormap not in COLORMAP_NAMES:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")
    return colormap


def normalize_request_metrics_and_outputs(
    metrics: str | None,
    outputs: str | None,
    *,
    normalize_metrics_mode: Callable[[str], str],
    parse_outputs: Callable[[str], list[str]],
) -> tuple[str, str]:
    """Normalize metrics/output fields via inference wrappers supplied by routes."""

    metrics_value = settings.DEPTHLENS_DEFAULT_METRICS if metrics is None else metrics
    outputs_value = settings.DEPTHLENS_DEFAULT_OUTPUTS if outputs is None else outputs
    normalized_metrics = normalize_metrics_mode(metrics_value)
    normalized_outputs = ",".join(parse_outputs(outputs_value))
    return str(normalized_metrics), normalized_outputs
