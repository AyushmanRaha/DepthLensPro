from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.model_registry import UnknownModelError, normalize_model_id, resolve_onnx_path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MiDaS Small", "midas_small"),
        ("MiDaS_small", "midas_small"),
        ("midas_small", "midas_small"),
        ("DPT Hybrid", "dpt_hybrid"),
        ("DPT_Hybrid", "dpt_hybrid"),
        ("dpt_hybrid", "dpt_hybrid"),
        ("DPT Large", "dpt_large"),
        ("DPT_Large", "dpt_large"),
        ("dpt_large", "dpt_large"),
    ],
)
def test_normalize_model_id_variants(raw: str, expected: str) -> None:
    assert normalize_model_id(raw) == expected


def test_normalize_unknown_model_raises_structured_error() -> None:
    with pytest.raises(UnknownModelError) as exc:
        normalize_model_id("not a model")
    assert exc.value.error_code == "UNKNOWN_MODEL"
    assert "midas_small" in exc.value.valid_models


def test_resolve_missing_onnx_read_mode_does_not_return_fake_path(tmp_path: Path) -> None:
    payload = resolve_onnx_path("midas_small", output_dir=tmp_path)
    assert payload["onnx_path"] is None
    assert Path(payload["expected_path"]).is_absolute()
    assert payload["exists"] is False
    assert payload["error"] == "missing_file"


def test_resolve_write_mode_returns_canonical_destination(tmp_path: Path) -> None:
    payload = resolve_onnx_path("midas_small", output_dir=tmp_path, for_write=True)
    assert payload["onnx_path"] == str((tmp_path / "midas_small.onnx").resolve())
    assert payload["expected_path"] == payload["onnx_path"]
    assert payload["exists"] is False
    assert payload["error"] == "missing_file"


def test_resolve_valid_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "midas_small.onnx"
    path.write_bytes(b"onnx")
    payload = resolve_onnx_path("MiDaS Small", output_dir=tmp_path)
    assert payload["model_id"] == "midas_small"
    assert payload["exists"] is True
    assert payload["size_bytes"] == 4
    assert payload["onnx_path"] == str(path.resolve())


def test_depthlenspro_model_dir_env_override(monkeypatch: Any, tmp_path: Path) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    expected = onnx_dir / "dpt_hybrid.onnx"
    expected.write_bytes(b"onnx")
    monkeypatch.setenv("DEPTHLENSPRO_MODEL_DIR", str(tmp_path))
    payload = resolve_onnx_path("DPT_Hybrid")
    assert payload["source"] == "env_model_dir"
    assert payload["onnx_path"] == str(expected.resolve())
    assert payload["exists"] is True


def test_onnx_weights_dir_env_override(monkeypatch: Any, tmp_path: Path) -> None:
    expected = tmp_path / "midas_small.onnx"
    expected.write_bytes(b"onnx")
    monkeypatch.delenv("DEPTHLENSPRO_MODEL_DIR", raising=False)
    monkeypatch.delenv("DEPTHLENS_ONNX_DIR", raising=False)
    monkeypatch.setenv("ONNX_WEIGHTS_DIR", str(tmp_path))
    payload = resolve_onnx_path("midas_small")
    assert payload["source"] == "env_weights_dir"
    assert payload["onnx_path"] == str(expected.resolve())


def test_export_and_inference_paths_share_canonical_location(tmp_path: Path) -> None:
    read_payload = resolve_onnx_path("dpt_large", output_dir=tmp_path)
    write_payload = resolve_onnx_path("dpt_large", output_dir=tmp_path, for_write=True)
    assert read_payload["expected_path"] == write_payload["onnx_path"]
