"""Route-level tests for the DepthLens Pro API."""

from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient

from backend.main import app
from backend.services import cache_service

client = TestClient(app)


def _png_bytes() -> bytes:
    return b"stub-image-bytes"


def test_health_checkpoint(monkeypatch: Any) -> None:
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

    def fake_process(
        raw: bytes, model: str, colormap: str, device: str, filename: str | None
    ) -> dict[str, Any]:
        return {
            "depth_map": "depth-png",
            "grayscale": "gray-png",
            "metrics": {"mean": 0.5},
            "latency_ms": 1.2,
            "model": model,
            "colormap": colormap,
            "device_used": device,
            "resolution": {"width": 8, "height": 8},
            "filename": filename,
            "cached": False,
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
