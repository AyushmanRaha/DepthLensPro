from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.main import app
from backend.services import cache_service
from backend.services.model_assets import ModelAssetsUnavailableError, model_assets_status

client = TestClient(app)


def test_model_assets_onnx_missing_not_fatal_with_midas_cache(monkeypatch: Any, tmp_path: Path) -> None:
    torch_home = tmp_path / "torch-cache"
    repo = torch_home / "hub" / "intel-isl_MiDaS_master"
    (repo / "midas").mkdir(parents=True)
    (repo / "hubconf.py").write_text("# fake")
    ckpt = torch_home / "hub" / "checkpoints"
    ckpt.mkdir(parents=True)
    for name in ("midas_v21_small_256.pt", "dpt_hybrid_384.pt", "dpt_large_384.pt"):
        (ckpt / name).write_bytes(b"fake")
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setenv("DEPTHLENS_ONNX_DIR", str(tmp_path / "missing-onnx"))
    monkeypatch.setenv("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", "1")

    payload = model_assets_status()

    assert payload["onnx"]["any_ready"] is False
    assert payload["pytorch_hub"]["midas_repo_cached"] is True
    assert payload["standard_build_ready"] is True
    assert payload["model_assets_ready"] is True


def test_model_assets_missing_offline_is_unavailable(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty-cache"))
    monkeypatch.setenv("DEPTHLENS_ONNX_DIR", str(tmp_path / "missing-onnx"))
    monkeypatch.setenv("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", "1")

    payload = model_assets_status()

    assert payload["model_assets_ready"] is False
    assert payload["inference_ready"] is False
    assert payload["fatal_reason"]


def test_ready_includes_model_asset_fields() -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert "runtime_imports_ready" in payload
    assert "model_assets_ready" in payload
    assert "inference_ready" in payload
    assert "model_assets" in payload


def test_estimate_model_assets_error_is_503(monkeypatch: Any) -> None:
    cache_service.clear()
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: ({"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}}, "cpu", {}),
    )

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ModelAssetsUnavailableError()

    monkeypatch.setattr("backend.api.routes.process_image", fail)
    response = client.post("/estimate", files={"file": ("x.png", io.BytesIO(b"asset-missing"), "image/png")})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error_code"] == "MODEL_ASSETS_UNAVAILABLE"
    assert "ONNX is optional" in detail["standard_build_note"]


def test_reconstruct_model_assets_error_is_503(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.routes._resolve", lambda requested: "cpu")
    monkeypatch.setattr(
        "backend.api.routes._cached_devices",
        lambda force=False: ({"cpu": {"name": "CPU", "type": "cpu", "compute_classes": ["cpu"], "available": True}}, "cpu", {}),
    )

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ModelAssetsUnavailableError()

    monkeypatch.setattr("backend.api.routes.reconstruct_point_cloud", fail)
    response = client.post("/api/reconstruct", files={"file": ("x.png", io.BytesIO(b"asset-missing-recon"), "image/png")})
    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "MODEL_ASSETS_UNAVAILABLE"
