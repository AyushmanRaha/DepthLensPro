"""Unit tests for thin API helper modules."""

from __future__ import annotations

import io
import logging
from typing import Any, cast

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from backend.api import device_state
from backend.api.errors import (
    benchmark_timeout,
    detector_unavailable,
    generic_inference_failure,
    reconstruction_timeout,
)
from backend.api.system_telemetry import disk_telemetry, percent, telemetry_status
from backend.api.validation import (
    normalize_request_colormap,
    normalize_request_metrics_and_outputs,
    validate_image_upload_content_type,
    validate_upload_size,
)


def test_validation_preserves_public_messages() -> None:
    upload = UploadFile(
        file=io.BytesIO(b"text"),
        filename="notes.txt",
        headers=Headers({"content-type": "text/plain"}),
    )
    with pytest.raises(HTTPException) as content_type_exc:
        validate_image_upload_content_type(upload)
    assert content_type_exc.value.status_code == 415
    assert content_type_exc.value.detail == "Expected an image file"

    with pytest.raises(HTTPException) as size_exc:
        validate_upload_size(b"1234", max_size_mb=0)
    assert size_exc.value.status_code == 413
    assert size_exc.value.detail == "Image file exceeds 0 MB limit"

    with pytest.raises(HTTPException) as colormap_exc:
        normalize_request_colormap("missing")
    assert colormap_exc.value.status_code == 422
    assert colormap_exc.value.detail == "Unknown colormap 'missing'"


def test_metrics_and_outputs_normalization_uses_supplied_wrappers() -> None:
    metrics, outputs = normalize_request_metrics_and_outputs(
        "full",
        "color,gray",
        normalize_metrics_mode=lambda value: value.upper(),
        parse_outputs=lambda value: value.split(","),
    )
    assert metrics == "FULL"
    assert outputs == "color,gray"


def test_error_helpers_preserve_payloads() -> None:
    assert cast(dict[str, str], benchmark_timeout().detail) == {
        "error_code": "BENCHMARK_TIMEOUT",
        "message": "Benchmark timed out · depth engine remains available",
    }
    assert cast(dict[str, str], reconstruction_timeout().detail) == {
        "error_code": "RECONSTRUCTION_TIMEOUT",
        "message": "Point cloud generation timed out · depth engine remains available",
    }
    assert cast(dict[str, str], generic_inference_failure().detail) == {
        "error_code": "INFERENCE_FAILED",
        "message": "Depth inference failed",
    }

    class DetectorUnavailableError(RuntimeError):
        error_code = "DETECTOR_WEIGHTS_UNAVAILABLE"

    detail = cast(
        dict[str, str], detector_unavailable(DetectorUnavailableError("weights missing")).detail
    )
    assert detail["error_code"] == "DETECTOR_WEIGHTS_UNAVAILABLE"
    assert detail["message"] == "weights missing"
    assert "Run setup" in detail["action"]


def test_device_state_falls_back_to_cpu_and_caches(caplog: pytest.LogCaptureFixture) -> None:
    device_state.DEVICE_CACHE.update(
        {"expires_at": 0.0, "devices": None, "primary": "cpu", "error": None}
    )
    calls = {"count": 0}

    def fail_devices() -> dict[str, Any]:
        calls["count"] += 1
        raise RuntimeError("probe failed")

    devs, primary, meta = device_state.cached_devices(
        available_devices=fail_devices,
        default_device_key=lambda: "cpu",
        log=logging.getLogger("test"),
    )
    second, _, second_meta = device_state.cached_devices(
        available_devices=fail_devices,
        default_device_key=lambda: "cpu",
        log=logging.getLogger("test"),
    )
    assert devs["cpu"]["available"] is True
    assert primary == "cpu"
    assert meta["error"] == "probe failed"
    assert second == devs
    assert second_meta["cached"] is True
    assert calls["count"] == 1


def test_validated_device_refreshes_once_for_stale_cache() -> None:
    calls = {"cache": 0, "resolve": 0}

    def cached_devices_func(force: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
        calls["cache"] += 1
        devices = {"cpu": {"available": True}, "cuda:0": {"available": True}}
        return devices, "cpu", {}

    def resolve(device: str) -> str:
        calls["resolve"] += 1
        if calls["resolve"] == 1:
            raise ValueError("stale")
        return "cuda:0"

    resolved = device_state.validated_device_or_422(
        "cuda:0",
        cached_devices_func=cached_devices_func,
        resolve=resolve,
        log=logging.getLogger("test"),
    )
    assert resolved == "cuda:0"
    assert calls == {"cache": 2, "resolve": 2}


def test_system_telemetry_status_and_disk_fields() -> None:
    assert percent(1, 4) == 25.0
    assert telemetry_status({"status": "ok"}, {"status": "degraded"}) == "degraded"
    disk = disk_telemetry("/")
    assert {"status", "path", "usage_percent", "limit_percent"}.issubset(disk)
