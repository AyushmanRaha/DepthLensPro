"""Route tests for the reconstruction API contract."""

from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _mock_cpu(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: (
            {"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}},
            "cpu",
            {},
        ),
    )
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")


def test_reconstruct_invalid_format_gives_422(monkeypatch: Any) -> None:
    _mock_cpu(monkeypatch)

    def fake_reconstruct(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("export_format must be one of: ply, obj")

    monkeypatch.setattr("backend.api.routes.reconstruct_point_cloud", fake_reconstruct)

    response = client.post(
        "/api/reconstruct",
        files={"file": ("sample.png", io.BytesIO(b"image-bytes"), "image/png")},
        data={"export_format": "glb"},
    )

    assert response.status_code == 422
    assert "export_format" in response.json()["detail"]


def test_reconstruct_rejects_non_image_upload(monkeypatch: Any) -> None:
    _mock_cpu(monkeypatch)

    response = client.post(
        "/api/reconstruct",
        files={"file": ("notes.txt", io.BytesIO(b"not image"), "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Expected an image file"


def test_reconstruct_rejects_oversized_upload(monkeypatch: Any) -> None:
    _mock_cpu(monkeypatch)

    response = client.post(
        "/api/reconstruct",
        files={"file": ("huge.png", io.BytesIO(b"x" * (21 * 1024 * 1024)), "image/png")},
    )

    assert response.status_code == 413
    assert "20 MB" in response.json()["detail"]


def test_reconstruct_happy_path_returns_artifact_contract(monkeypatch: Any) -> None:
    _mock_cpu(monkeypatch)

    def fake_reconstruct(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "filename": kwargs["filename"],
            "artifact_filename": "sample_point_cloud.ply",
            "artifact_format": kwargs["export_format"],
            "artifact_mime": "model/ply",
            "artifact_base64": "cGx5Cg==",
            "artifact_size_bytes": 4,
            "preview": {
                "points": [[0.0, 0.0, 0.0, 255, 255, 255]],
                "point_count": 1,
                "truncated": False,
            },
            "depth_map": "png-base64",
            "reconstruction": {"point_count": 1},
            "resolution": {"width": 1, "height": 1},
            "model": kwargs["model"],
            "model_id": kwargs["model"],
            "model_display_name": "MiDaS Small",
            "device_used": kwargs["device"],
            "engine_used": "cache",
            "fallback_used": False,
            "depth_cached": True,
            "latency_ms": 0.0,
            "total_latency_ms": 1.0,
            "warnings": [],
        }

    monkeypatch.setattr("backend.api.routes.reconstruct_point_cloud", fake_reconstruct)

    response = client.post(
        "/reconstruct",
        files={"file": ("sample.png", io.BytesIO(b"image-bytes"), "image/png")},
        data={"export_format": "ply", "preview_points": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["artifact_filename"].endswith(".ply")
    assert payload["artifact_base64"]
    assert payload["preview"]["point_count"] == 1
