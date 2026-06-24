"""Safety-contract tests for behavior-preserving refactor phases."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.model_metadata import COLORMAP_NAMES
from backend.model_registry import MODEL_REGISTRY

client = TestClient(app)


def test_required_public_routes_still_exist() -> None:
    required_routes = {
        "/",
        "/live",
        "/ready",
        "/health",
        "/devices",
        "/models",
        "/colormaps",
        "/onnx/status",
        "/benchmark",
        "/api/benchmark",
        "/cache/metrics",
        "/estimate",
        "/batch",
        "/compare",
        "/api/compare",
        "/api/reconstruct",
        "/reconstruct",
        "/api/detect",
        "/detect",
    }

    registered_paths = {route.path for route in app.routes}

    assert required_routes <= registered_paths


def test_live_contract_remains_lightweight_shape() -> None:
    response = client.get("/live")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "service",
        "status",
        "version",
        "pid",
        "timestamp",
        "uptime_seconds",
    } <= set(payload)
    assert payload["service"] == "DepthLens Pro API"
    assert payload["status"] == "ok"
    assert isinstance(payload["version"], str)
    assert isinstance(payload["pid"], int)
    assert isinstance(payload["timestamp"], float)
    assert isinstance(payload["uptime_seconds"], float)


def test_models_return_all_canonical_model_ids() -> None:
    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    returned_ids = {model["model_id"] for model in payload["models"]}
    assert returned_ids == set(MODEL_REGISTRY)
    assert returned_ids == {"midas_small", "dpt_hybrid", "dpt_large"}


def test_colormaps_include_current_supported_set() -> None:
    response = client.get("/colormaps")

    assert response.status_code == 200
    payload = response.json()
    assert set(COLORMAP_NAMES) <= set(payload["colormaps"])
    assert set(payload["colormaps"]) == {
        "inferno",
        "plasma",
        "viridis",
        "magma",
        "jet",
        "hot",
        "bone",
        "turbo",
    }
