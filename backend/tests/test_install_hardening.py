from __future__ import annotations

import json
from pathlib import Path

from backend.model_registry import get_model_spec, resolve_onnx_path
from backend.scripts import export_onnx


def test_model_registry_onnx_input_sizes() -> None:
    assert get_model_spec("midas_small").input_size == (256, 256)
    assert get_model_spec("dpt_hybrid").input_size == (384, 384)
    assert get_model_spec("dpt_large").input_size == (384, 384)


def test_resolve_onnx_path_repo_and_packaged_context(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "Resources" / "models"
    onnx = model_dir / "onnx"
    onnx.mkdir(parents=True)
    target = onnx / "midas_small.onnx"
    target.write_bytes(b"fake")
    monkeypatch.setenv("DEPTHLENSPRO_MODEL_DIR", str(model_dir))
    payload = resolve_onnx_path("midas_small")
    assert payload["onnx_path"] == str(target.resolve())
    assert payload["source"] == "env_model_dir"


def test_validate_only_detects_invalid_file(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "midas_small.onnx"
    path.write_bytes(b"not onnx")
    monkeypatch.setattr(export_onnx, "create_onnx_session", lambda *a, **k: {"ok": False, "error_code": "ONNX_SESSION_INIT_FAILED", "message": "bad"})
    result = export_onnx.validate_model("midas_small", tmp_path)
    assert result["ok"] is False
    assert result["error_code"] in {"INVALID_CHECKER", "INVALID_SESSION"}


def test_electron_extra_resources_include_models() -> None:
    data = json.loads(Path("electron-app/package.json").read_text())
    assert any(item.get("from") == "../models" and item.get("to") == "models" for item in data["build"]["extraResources"])


def test_readme_mentions_setup_scripts() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "scripts/setup-macos.sh" in readme
    assert "scripts/build-native-macos.sh" in readme
