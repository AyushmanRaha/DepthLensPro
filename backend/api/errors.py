"""HTTP exception helpers used by route orchestration."""

from __future__ import annotations

import logging

from fastapi import HTTPException

from backend.services.model_assets import ModelAssetsUnavailableError

BENCHMARK_TIMEOUT_MESSAGE = "Benchmark timed out · depth engine remains available"
RECONSTRUCTION_TIMEOUT_MESSAGE = "Point cloud generation timed out · depth engine remains available"


def inference_dependency_unavailable(exc: Exception, log: logging.Logger) -> HTTPException:
    log.warning("Inference runtime dependency unavailable: %s: %s", type(exc).__name__, exc)
    return HTTPException(
        503,
        {
            "error_code": "INFERENCE_RUNTIME_UNAVAILABLE",
            "message": "Inference runtime is not ready. Check /ready for dependency diagnostics.",
        },
    )


def model_assets_unavailable(exc: ModelAssetsUnavailableError) -> HTTPException:
    return HTTPException(503, exc.to_payload())


def generic_inference_failure(message: str = "Depth inference failed") -> HTTPException:
    return HTTPException(500, {"error_code": "INFERENCE_FAILED", "message": message})


def benchmark_timeout() -> HTTPException:
    return HTTPException(
        504,
        {"error_code": "BENCHMARK_TIMEOUT", "message": BENCHMARK_TIMEOUT_MESSAGE},
    )


def reconstruction_timeout() -> HTTPException:
    return HTTPException(
        504,
        {"error_code": "RECONSTRUCTION_TIMEOUT", "message": RECONSTRUCTION_TIMEOUT_MESSAGE},
    )


def detector_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        503,
        {
            "error_code": getattr(exc, "error_code", "DETECTOR_UNAVAILABLE"),
            "message": str(exc) or "Local object detector is unavailable",
            "action": (
                "Run setup to install detector dependencies, or retry when network "
                "access is available for lazy weights download."
            ),
        },
    )


def generic_detector_failure() -> HTTPException:
    return HTTPException(
        500, {"error_code": "DETECTION_FAILED", "message": "Object detection failed"}
    )


def generic_reconstruction_failure() -> HTTPException:
    return HTTPException(
        500, {"error_code": "RECONSTRUCTION_FAILED", "message": "Point cloud generation failed"}
    )
