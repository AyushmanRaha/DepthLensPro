from __future__ import annotations

import io
import json
from typing import Any

from fastapi.testclient import TestClient

from backend.main import app
from backend.services import cache_service, observability

client = TestClient(app)


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _cpu(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: ({"cpu": {"name": "CPU", "type": "cpu", "available": True}}, "cpu", {}),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")


def setup_function() -> None:
    observability.reset_for_tests()
    cache_service.clear()


def test_observability_snapshot_keys() -> None:
    response = client.get("/api/observability")
    assert response.status_code == 200
    payload = response.json()
    for key in [
        "status",
        "enabled",
        "prometheus_enabled",
        "process",
        "http",
        "inference",
        "cache",
        "traces",
        "crashes",
        "benchmarks",
    ]:
        assert key in payload


def test_observability_alias_and_metrics_text() -> None:
    assert client.get("/observability").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "text/plain" in metrics.headers["content-type"]
    if observability.prometheus_enabled():
        assert "depthlens_" in metrics.text


def test_reset_for_tests_clears_histories() -> None:
    observability.record_inference("MiDaS_small", "pytorch", "cpu", 12)
    assert observability.snapshot()["inference"]["total"] == 1
    observability.reset_for_tests()
    snap = observability.snapshot()
    assert snap["inference"]["total"] == 0
    assert snap["traces"]["recent"] == []
    assert snap["benchmarks"]["history"] == []


def test_mocked_estimate_success_records_inference_without_filename(monkeypatch: Any) -> None:
    _cpu(monkeypatch)

    def fake_process(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "depth_map": "x",
            "metrics": {},
            "latency_ms": 2.5,
            "model": "MiDaS_small",
            "model_id": "MiDaS_small",
            "device_used": "cpu",
            "engine_used": "pytorch",
            "resolution": {"width": 2, "height": 3},
            "filename": "secret-name.png",
            "cached": False,
            "outputs": ["color"],
        }

    monkeypatch.setattr("backend.api.routes.process_image", fake_process)
    response = client.post(
        "/estimate",
        files={"file": ("secret-name.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"device": "auto"},
    )
    assert response.status_code == 200
    snap_text = json.dumps(client.get("/api/observability").json())
    assert '"total": 1' in snap_text
    assert "secret-name.png" not in snap_text


def test_mocked_estimate_failure_sanitizes_path_and_keeps_response(monkeypatch: Any) -> None:
    _cpu(monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("failed loading /tmp/private/model.onnx for upload secret-fail.png")

    monkeypatch.setattr("backend.api.routes.process_image", fail)
    response = client.post(
        "/estimate",
        files={"file": ("secret-fail.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"device": "auto"},
    )
    assert response.status_code == 500
    assert response.json() == {
        "detail": {"error_code": "INFERENCE_FAILED", "message": "Depth inference failed"}
    }
    snap_text = json.dumps(client.get("/api/observability").json())
    assert "/tmp/private/model.onnx" not in snap_text
    assert "secret-fail.png" not in snap_text
    assert "[path]" in snap_text


def test_mocked_benchmark_records_history(monkeypatch: Any) -> None:
    def fake_benchmark(
        model: str, device: str, iterations: int, engine: str = "both"
    ) -> dict[str, Any]:
        observability.record_benchmark(
            model,
            "MiDaS Small",
            device,
            "cpu",
            iterations,
            10.0,
            5.0,
            2.0,
            "ok",
            "CPUExecutionProvider",
            0,
            "ok",
        )
        return {
            "model": model,
            "model_id": model,
            "display_name": "MiDaS Small",
            "device_requested": device,
            "device_resolved": "cpu",
            "iterations": iterations,
            "engine_requested": engine,
            "results": [],
            "comparison": {},
            "speedup": 2.0,
            "pytorch": {"latency_ms": 10.0},
            "onnx": {"latency_ms": 5.0, "status": "ok"},
            "warnings": [],
        }

    monkeypatch.setattr("backend.api.routes.run_benchmark", fake_benchmark)
    response = client.get("/api/benchmark?model=MiDaS_small&device=auto&iterations=2")
    assert response.status_code == 200
    assert client.get("/api/observability").json()["benchmarks"]["history"]


def test_cache_metrics_records_event() -> None:
    response = client.get("/cache/metrics")
    assert response.status_code == 200
    events = client.get("/api/observability").json()["cache"]["events"]
    assert events.get("metrics", 0) >= 1
