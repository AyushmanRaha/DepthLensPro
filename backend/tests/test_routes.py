"""Route-level tests for the DepthLens Pro API."""

from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient

from backend.api.routes import process_image_async
from backend.main import app
from backend.services import cache_service

client = TestClient(app)


def _png_bytes() -> bytes:
    return b"stub-image-bytes"


def _stable_health_diagnostics(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.routes._memory_telemetry", lambda: {"status": "ok"})
    monkeypatch.setattr("backend.api.routes._disk_telemetry", lambda: {"status": "ok"})
    monkeypatch.setattr(
        "backend.api.routes.onnx_status_payload",
        lambda device: {"overall_status": "onnx_unavailable"},
    )
    monkeypatch.setattr(
        "backend.api.routes._cached_readiness_payload",
        lambda device: (
            {"overall_status": "pytorch_ready_onnx_unavailable", "models": {}},
            {"cached": False},
        ),
    )


def test_health_checkpoint(monkeypatch: Any) -> None:
    _stable_health_diagnostics(monkeypatch)
    monkeypatch.setattr(
        "backend.api.routes._available_devices",
        lambda: {
            "cpu": {
                "name": "CPU · test",
                "hardware_name": "test",
                "type": "cpu",
                "compute_classes": ["cpu"],
                "available": True,
            }
        },
    )
    monkeypatch.setattr("backend.api.routes._default_device_key", lambda: "cpu")
    monkeypatch.setattr(
        "backend.api.routes._acceleration_checks",
        lambda devs: {
            "cuda": {"available": False, "operational": False},
            "mps": {"available": False, "operational": False},
            "xpu": {"available": False, "operational": False},
        },
    )

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["primary_device"] == "cpu"
    assert payload["devices"]["cpu"]["available"] is True


def test_static_metadata_routes() -> None:
    assert client.get("/").json()["service"] == "DepthLens Pro API"
    assert client.get("/models").status_code == 200
    assert "inferno" in client.get("/colormaps").json()["colormaps"]


def test_cache_metrics_route_reports_dashboard_fields() -> None:
    cache_service.clear()

    response = client.get("/cache/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert {"total_hits", "cache_misses", "keyspace_size"}.issubset(payload)
    assert payload["ttl_seconds"] == cache_service.CACHE_TTL_SECONDS


def test_estimate_uses_mocked_processing_and_cache(monkeypatch: Any) -> None:
    cache_service.clear()
    monkeypatch.setattr(
        "backend.api.routes._available_devices",
        lambda: {
            "cpu": {
                "name": "CPU · test",
                "hardware_name": "test",
                "type": "cpu",
                "compute_classes": ["cpu"],
                "available": True,
            }
        },
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")

    calls = {"count": 0}

    def fake_process(
        raw: bytes,
        model: str,
        colormap: str,
        device: str,
        filename: str | None,
        metrics: str | None = None,
        outputs: str | None = None,
        max_dim: int | None = None,
        *args: Any,
    ) -> dict[str, Any]:
        calls["count"] += 1
        return {
            "depth_map": "depth-png",
            "grayscale": "gray-png",
            "metrics": {} if metrics == "none" else {"mean": 0.5, "mode": metrics},
            "latency_ms": 1.2,
            "model": model,
            "colormap": colormap,
            "device_used": device,
            "resolution": {"width": 8, "height": 8},
            "filename": filename,
            "cached": False,
            "outputs": (outputs or "color").split(","),
            **({"depth_map": "depth-png"} if outputs in (None, "color", "color,gray") else {}),
            **({"grayscale": "gray-png"} if outputs in ("gray", "color,gray") else {}),
        }

    monkeypatch.setattr("backend.api.routes.process_image", fake_process)

    files = {"file": ("sample.png", io.BytesIO(_png_bytes()), "image/png")}
    response = client.post(
        "/estimate",
        files=files,
        data={"model": "MiDaS_small", "colormap": "inferno", "device": "auto"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["depth_map"] == "depth-png"
    assert payload["cached"] is False

    second = client.post(
        "/estimate",
        files={"file": ("sample.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"model": "MiDaS_small", "colormap": "inferno", "device": "auto"},
    )
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert calls["count"] == 1


def test_estimate_generic_failure_uses_sanitized_error(monkeypatch: Any) -> None:
    cache_service.clear()
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}},
            "cpu",
            {},
        ),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")

    def fail_process(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("secret path /tmp/model.onnx")

    monkeypatch.setattr("backend.api.routes.process_image", fail_process)

    response = client.post(
        "/estimate",
        files={"file": ("sample.png", io.BytesIO(b"unique-failure-image"), "image/png")},
        data={"device": "auto"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error_code": "INFERENCE_FAILED",
        "message": "Depth inference failed",
    }


def test_batch_rejects_non_image_upload_before_processing(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}},
            "cpu",
            {},
        ),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")

    def fail_if_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("non-image batch item should not reach inference")

    monkeypatch.setattr("backend.api.routes.process_image", fail_if_called)

    response = client.post(
        "/batch",
        files=[("files", ("notes.txt", io.BytesIO(b"not image"), "text/plain"))],
        data={"device": "auto"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["succeeded"] == 0
    assert payload["failed"] == 1
    assert payload["errors"][0]["error"] == "Expected an image file"


def test_benchmark_route_uses_service(monkeypatch: Any) -> None:
    def fake_benchmark(model: str, device: str, iterations: int) -> dict[str, Any]:
        return {
            "model": model,
            "device_requested": device,
            "iterations": iterations,
            "results": [],
            "comparison": {},
        }

    monkeypatch.setattr("backend.api.routes.run_benchmark", fake_benchmark)

    response = client.get("/api/benchmark?model=MiDaS_small&device=auto&iterations=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "MiDaS_small"
    assert payload["iterations"] == 2


def test_live_is_lightweight(monkeypatch: Any) -> None:
    def fail_devices() -> None:
        raise AssertionError("/live must not discover devices")

    monkeypatch.setattr("backend.api.routes._available_devices", fail_devices)
    response = client.get("/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "DepthLens Pro API"
    assert "pid" in payload


def test_devices_always_include_cpu_on_discovery_failure(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.api.routes._DEVICE_CACHE",
        {"expires_at": 0.0, "devices": None, "primary": "cpu", "error": None},
    )

    def fail_devices() -> None:
        raise RuntimeError("probe failed")

    monkeypatch.setattr("backend.api.routes._available_devices", fail_devices)
    response = client.get("/devices")

    assert response.status_code == 200
    payload = response.json()
    assert payload["devices"]["cpu"]["available"] is True
    assert payload["primary_device"] == "cpu"


def test_health_stays_ok_when_optional_acceleration_probe_fails(monkeypatch: Any) -> None:
    _stable_health_diagnostics(monkeypatch)
    monkeypatch.setattr(
        "backend.api.routes._DEVICE_CACHE",
        {"expires_at": 0.0, "devices": None, "primary": "cpu", "error": None},
    )
    monkeypatch.setattr(
        "backend.api.routes._ACCEL_CACHE", {"expires_at": 0.0, "checks": None, "error": None}
    )
    monkeypatch.setattr(
        "backend.api.routes._available_devices",
        lambda: {
            "cpu": {
                "name": "CPU · test",
                "hardware_name": "test",
                "type": "cpu",
                "compute_classes": ["cpu"],
                "available": True,
            },
            "mps": {
                "name": "GPU · test",
                "hardware_name": "test",
                "type": "mps",
                "available": True,
            },
        },
    )
    monkeypatch.setattr("backend.api.routes._default_device_key", lambda: "mps")

    def fail_accel(devs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("mps probe failed")

    monkeypatch.setattr("backend.api.routes._acceleration_checks", fail_accel)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["acceleration_checks"]["mps"]["operational"] is False
    assert "timings_ms" in payload


def test_estimate_metrics_and_outputs_modes(monkeypatch: Any) -> None:
    cache_service.clear()
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {
                "cpu": {
                    "name": "CPU · test",
                    "hardware_name": "test",
                    "type": "cpu",
                    "compute_classes": ["cpu"],
                    "available": True,
                }
            },
            "cpu",
            {"cached": False},
        ),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")

    def fake_process(
        raw: bytes,
        model: str,
        colormap: str,
        device: str,
        filename: str | None,
        metrics: str | None = None,
        outputs: str | None = None,
        max_dim: int | None = None,
        *args: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metrics": {} if metrics == "none" else {"mode": metrics},
            "latency_ms": 1.0,
            "model": model,
            "colormap": colormap,
            "device_used": device,
            "resolution": {"width": 4, "height": 4},
            "filename": filename,
            "cached": False,
        }
        if outputs in {"color", "color,gray"}:
            payload["depth_map"] = "depth-png"
        if outputs in {"gray", "color,gray"}:
            payload["grayscale"] = "gray-png"
        return payload

    monkeypatch.setattr("backend.api.routes.process_image", fake_process)
    cases = [
        ({"metrics": "none", "outputs": "color"}, "depth_map", "grayscale", {}),
        ({"metrics": "fast", "outputs": "gray"}, "grayscale", "depth_map", {"mode": "fast"}),
        ({"metrics": "full", "outputs": "color,gray"}, "depth_map", None, {"mode": "full"}),
    ]
    for data, present, absent, expected_metrics in cases:
        response = client.post(
            "/estimate",
            files={
                "file": ("sample.png", io.BytesIO(_png_bytes() + present.encode()), "image/png")
            },
            data={"model": "MiDaS_small", "colormap": "inferno", "device": "auto", **data},
        )
        assert response.status_code == 200
        payload = response.json()
        assert present in payload
        if absent:
            assert absent not in payload
        assert payload["metrics"] == expected_metrics


def test_estimate_gt_required_without_file_returns_clear_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}},
            "cpu",
            {},
        ),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")
    response = client.post(
        "/estimate",
        files={"file": ("sample.png", io.BytesIO(_png_bytes()), "image/png")},
        data={
            "model": "MiDaS_small",
            "colormap": "inferno",
            "device": "auto",
            "gt_required": "true",
        },
    )
    assert response.status_code == 422
    assert "GT depth file" in response.json()["detail"]


def test_onnx_status_route_reports_paths_and_providers() -> None:
    response = client.get("/onnx/status?device=cpu")
    assert response.status_code == 200
    payload = response.json()
    assert "midas_small" in payload["supported_model_ids"]
    small = payload["models"]["midas_small"]
    assert small["expected_path"].endswith("midas_small.onnx")
    assert "runtime" in small
    assert "available_providers" in payload["runtime"]


def test_upload_size_limit_accepts_exact_limit_and_rejects_one_byte_over(monkeypatch: Any) -> None:
    cache_service.clear()
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}},
            "cpu",
            {},
        ),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")
    monkeypatch.setattr("backend.api.routes.MAX_UPLOAD_SIZE_MB", 1)
    monkeypatch.setattr(
        "backend.api.routes.process_image",
        lambda *args, **kwargs: {
            "metrics": {},
            "latency_ms": 0.1,
            "model": args[1],
            "colormap": args[2],
            "device_used": args[3],
            "resolution": {"width": 1, "height": 1},
            "filename": args[4],
            "cached": False,
        },
    )

    exact = client.post(
        "/estimate",
        files={"file": ("exact.png", io.BytesIO(b"a" * (1024 * 1024)), "image/png")},
        data={"device": "auto"},
    )
    over = client.post(
        "/estimate",
        files={"file": ("over.png", io.BytesIO(b"a" * (1024 * 1024 + 1)), "image/png")},
        data={"device": "auto"},
    )

    assert exact.status_code == 200
    assert over.status_code == 413


def test_unavailable_requested_device_returns_422_not_500(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}},
            "cpu",
            {},
        ),
    )

    response = client.post(
        "/estimate",
        files={"file": ("sample.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"device": "cuda:0"},
    )

    assert response.status_code == 422
    assert "unavailable" in str(response.json()["detail"])


def test_stale_cached_device_resolve_failure_returns_422(monkeypatch: Any) -> None:
    calls: list[bool] = []

    def fake_cached_devices(force: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
        calls.append(force)
        devices = {
            "cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}
        }
        if not force:
            devices["cuda:0"] = {"name": "GPU", "type": "cuda", "available": True}
        return devices, "cpu", {}

    monkeypatch.setattr("backend.api.routes._cached_devices", fake_cached_devices)
    monkeypatch.setattr(
        "backend.api.routes._resolve",
        lambda requested: (_ for _ in ()).throw(ValueError("Device 'cuda:0' unavailable")),
    )

    response = client.post(
        "/estimate",
        files={"file": ("sample.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"device": "cuda:0"},
    )

    assert response.status_code == 422
    assert calls == [False, True]


def test_route_async_offload_allows_two_concurrent_requests(monkeypatch: Any) -> None:
    import asyncio
    import threading
    import time

    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_process(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"latency_ms": 0.1}

    monkeypatch.setattr("backend.api.routes.process_image", slow_process)

    async def run_two() -> None:
        await asyncio.gather(
            process_image_async(b"a", "midas_small", "inferno", "cpu", "a.png"),
            process_image_async(b"b", "midas_small", "inferno", "cpu", "b.png"),
        )

    asyncio.run(run_two())

    assert max_active == 2


def test_detect_rejects_non_image_upload_before_service_call(monkeypatch: Any) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("detector should not be called")

    monkeypatch.setattr("backend.api.routes.detect_objects", fail_if_called)
    response = client.post(
        "/api/detect",
        files={"file": ("notes.txt", b"not image", "text/plain")},
    )

    assert response.status_code == 415


def test_detect_uses_monkeypatched_detector(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")

    def fake_detect(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["device"] == "cpu"
        assert kwargs["threshold"] == 0.35
        return {
            "detections": [{"label": "cup", "score": 0.91, "box": [1, 2, 3, 4]}],
            "model": "fake-detector",
            "device_used": "cpu",
            "latency_ms": 1.2,
            "resolution": {"width": 8, "height": 8},
        }

    monkeypatch.setattr("backend.api.routes.detect_objects", fake_detect)
    response = client.post(
        "/api/detect",
        files={"file": ("frame.jpg", _png_bytes(), "image/jpeg")},
        data={"threshold": "0.35", "max_detections": "3"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detections"][0]["label"] == "cup"
    assert payload["device_used"] == "cpu"


def test_detect_invalid_params_return_422(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")
    files = {"file": ("frame.jpg", _png_bytes(), "image/jpeg")}

    assert client.post("/api/detect", files=files, data={"threshold": "0.01"}).status_code == 422
    assert client.post("/api/detect", files=files, data={"max_detections": "21"}).status_code == 422


def test_detect_generic_failure_is_sanitized(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")

    def fail_detect(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("/tmp/private/model/path failed")

    monkeypatch.setattr("backend.api.routes.detect_objects", fail_detect)
    response = client.post(
        "/api/detect",
        files={"file": ("frame.jpg", _png_bytes(), "image/jpeg")},
    )

    assert response.status_code == 500
    payload = response.json()["detail"]
    assert payload == {"error_code": "DETECTION_FAILED", "message": "Object detection failed"}
    assert "private" not in str(payload)
