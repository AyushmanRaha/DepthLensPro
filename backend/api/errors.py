"""HTTP exception helpers and public error envelopes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from backend.services.model_assets import ModelAssetsUnavailableError
from backend.services.observability import sanitize_message

BENCHMARK_TIMEOUT_MESSAGE = "Benchmark timed out · depth engine remains available"
RECONSTRUCTION_TIMEOUT_MESSAGE = "Point cloud generation timed out · depth engine remains available"


def _sanitize_public_payload_value(key: str, value: Any) -> Any:
    if isinstance(value, str) and key in {"message", "action", "remediation"}:
        return sanitize_message(value)
    return value


def _without_private_public_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _sanitize_public_payload_value(key, value)
        for key, value in payload.items()
        if key != "retryable"
    }


def error_envelope(
    error_code: str,
    message: str,
    *,
    remediation: str | None = None,
    field: str | None = None,
    retryable: bool | None = None,
    **extra: Any,
) -> dict[str, Any]:
    del retryable  # accepted for caller compatibility, never exposed publicly
    payload: dict[str, Any] = {"error_code": error_code, "message": sanitize_message(message)}
    if remediation:
        payload["remediation"] = sanitize_message(remediation)
    if field:
        payload["field"] = field
    payload.update(_without_private_public_keys(extra))
    return _without_private_public_keys(payload)


def http_error(status_code: int, error_code: str, message: str, **kwargs: Any) -> HTTPException:
    return HTTPException(status_code, error_envelope(error_code, message, **kwargs))


def embedded_error(error_code: str, message: str, **kwargs: Any) -> dict[str, Any]:
    return error_envelope(error_code, message, **kwargs)


def validation_error(
    detail: str | dict[str, Any], *, field: str | None = None, code: str = "VALIDATION_ERROR"
) -> HTTPException:
    if isinstance(detail, str):
        return HTTPException(422, sanitize_message(detail))
    payload = _without_private_public_keys(dict(detail))
    payload.setdefault("error_code", code)
    if field is not None:
        payload["field"] = field
    return HTTPException(422, payload)


def inference_dependency_unavailable(exc: Exception, log: logging.Logger) -> HTTPException:
    log.warning(
        "Inference runtime dependency unavailable: %s: %s",
        type(exc).__name__,
        sanitize_message(exc),
    )
    return http_error(
        503,
        "INFERENCE_RUNTIME_UNAVAILABLE",
        "Inference runtime is not ready. Check /ready for dependency diagnostics.",
        retryable=True,
    )


def model_assets_unavailable(exc: ModelAssetsUnavailableError) -> HTTPException:
    payload = _without_private_public_keys(dict(exc.to_payload()))
    payload.setdefault("error_code", "MODEL_ASSET_UNAVAILABLE")
    payload.setdefault("message", "Model assets unavailable")
    return HTTPException(503, payload)


def generic_inference_failure(message: str = "Depth inference failed") -> HTTPException:
    return http_error(500, "INFERENCE_FAILED", message, retryable=True)


def timeout_error(
    route: str, message: str = "Request timed out · depth engine remains available"
) -> HTTPException:
    return http_error(504, "REQUEST_TIMEOUT", message, field=route, retryable=True)


def benchmark_timeout() -> HTTPException:
    return http_error(504, "BENCHMARK_TIMEOUT", BENCHMARK_TIMEOUT_MESSAGE, retryable=True)


def reconstruction_timeout() -> HTTPException:
    return http_error(504, "RECONSTRUCTION_TIMEOUT", RECONSTRUCTION_TIMEOUT_MESSAGE, retryable=True)


def detector_unavailable(exc: Exception) -> HTTPException:
    return http_error(
        503,
        str(getattr(exc, "error_code", "DETECTOR_UNAVAILABLE")),
        str(exc) or "Local object detector is unavailable",
        action=(
            "Run setup to install detector dependencies or retry after detector weights "
            "are available."
        ),
        retryable=True,
    )


def generic_detector_failure() -> HTTPException:
    return http_error(500, "DETECTION_FAILED", "Object detection failed", retryable=True)


def generic_reconstruction_failure() -> HTTPException:
    return http_error(500, "RECONSTRUCTION_FAILED", "Point cloud generation failed", retryable=True)
