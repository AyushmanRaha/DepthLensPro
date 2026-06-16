from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.services.ground_truth import compute_ground_truth_metrics

client = TestClient(app)


def test_gt_extended_metrics_are_computed() -> None:
    y, x = np.indices((16, 16), dtype=np.float32)
    gt = x + y + 1.0
    pred = (gt / gt.max()).astype(np.float32)
    result = compute_ground_truth_metrics(pred, gt, metadata={"warnings": []})
    metrics = result["metrics"]
    assert "gt_psnr" in metrics
    assert "gt_ssim" in metrics
    assert "surface_normal_error" in metrics
    assert "ordinal_error" in metrics
    assert result["unavailable"]["lpips"] == "optional_dependency_missing"


def test_models_status_and_capabilities_routes(monkeypatch: Any) -> None:
    fake = {
        "status": "setup_incomplete",
        "onnx_all_valid": False,
        "pytorch_all_visible": False,
        "models": {},
        "assets": [],
    }
    monkeypatch.setattr("backend.services.model_assets.model_status", lambda **kwargs: fake)
    monkeypatch.setattr(
        "backend.services.object_detection.detector_status",
        lambda device="auto": {"available": False, "last_error": "mock"},
    )
    response = client.get("/api/models/status")
    assert response.status_code == 200
    assert response.json()["status"] == "setup_incomplete"
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "supported_matrix" in payload
    assert "persistent_storage" in payload


def test_detect_status_route(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.services.object_detection.detector_status",
        lambda device="auto": {"available": True, "device_selected": device},
    )
    response = client.get("/api/detect/status?device=cpu")
    assert response.status_code == 200
    assert response.json()["available"] is True


def test_estimate_invalid_metrics_uses_estimate_error_code() -> None:
    image = Image.new("RGB", (4, 4), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    response = client.post(
        "/estimate", files={"file": ("x.png", buf.getvalue(), "image/png")}, data={"metrics": "bad"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INVALID_METRICS_MODE"
