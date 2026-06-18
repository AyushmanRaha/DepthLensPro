from pathlib import Path
from typing import Any

import pytest

from backend.services.model_assets import ModelAssetsUnavailableError, inspect_model_assets


def _fake_cache(root: Path) -> None:
    repo = root / "hub" / "intel-isl_MiDaS_master"
    (repo / "midas").mkdir(parents=True)
    (repo / "hubconf.py").write_text("# fake", encoding="utf-8")
    ckpt = root / "hub" / "checkpoints"
    ckpt.mkdir(parents=True)
    for name in ["midas_v21_small_256.pt", "dpt_hybrid_384.pt", "dpt_large_384.pt"]:
        (ckpt / name).write_bytes(b"x")


def test_model_asset_status_detects_missing_torch_cache(tmp_path: Path) -> None:
    status = inspect_model_assets(cache_root=tmp_path / "missing")
    assert status["pytorch_hub_cache_ready"] is False
    assert status["midas_repo_cached"] is False
    assert status["fatal_reason"] == "pytorch_midas_cache_missing_or_incomplete"


def test_model_asset_status_detects_fake_valid_cache(tmp_path: Path) -> None:
    cache = tmp_path / "torch-cache"
    _fake_cache(cache)
    status = inspect_model_assets(cache_root=cache)
    assert status["pytorch_hub_cache_ready"] is True
    assert status["checkpoint_summary"]["by_model"]["dpt_large"]["ready"] is True


def test_readiness_payload_inference_not_ready_when_assets_missing(
    monkeypatch: Any, tmp_path: Path
) -> None:
    pytest.importorskip("pydantic")
    pytest.importorskip("pydantic_settings")
    from backend.services import diagnostics

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "missing"))
    monkeypatch.setenv("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", "1")
    payload = diagnostics.readiness_payload()
    assert payload["backend_alive"] is True
    assert payload["runtime_imports_ready"] is True
    assert payload["model_assets_ready"] is False
    assert payload["inference_ready"] is False
    assert payload["fatal_reason"] == "pytorch_midas_cache_missing_or_incomplete"


def test_model_assets_error_payload() -> None:
    exc = ModelAssetsUnavailableError(
        status={"pytorch_hub_cache_path": "/tmp/cache", "recommended_action": "Run setup"}
    )
    payload = exc.to_payload()
    assert payload["error_code"] == "MODEL_ASSETS_UNAVAILABLE"
    assert payload["torch_home"] == "/tmp/cache"
    assert "PyTorch MiDaS assets are required" in payload["standard_build_note"]
