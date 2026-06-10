"""Benchmark service behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.services import benchmarks


def _missing_diag(device: str = "cpu") -> dict[str, Any]:
    return {
        "model": "MiDaS_small",
        "expected_path": "/tmp/MiDaS_small.onnx",
        "exists": False,
        "state": "missing_weights",
        "recommended_export_command": "python backend/scripts/export_onnx.py --model MiDaS_small",
        "runtime": {"requested_device": device},
    }


def test_ensure_onnx_weights_auto_exports_missing_model(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_status(model: str, device: str = "auto") -> dict[str, Any]:
        if calls:
            payload = _missing_diag(device)
            payload.update({"exists": True, "state": "available"})
            return payload
        return _missing_diag(device)

    def fake_export(model: str, output_dir: Path) -> Path:
        calls.append((model, output_dir))
        return output_dir / f"{model}.onnx"

    monkeypatch.setattr(benchmarks, "onnx_model_status", fake_status)
    monkeypatch.setattr(benchmarks, "onnx_model_path", lambda model: tmp_path / f"{model}.onnx")
    monkeypatch.setattr("backend.scripts.export_onnx.export_model", fake_export)

    result = benchmarks._ensure_onnx_weights("MiDaS_small", _missing_diag("cpu"))

    assert result["state"] == "available"
    assert calls == [("MiDaS_small", tmp_path)]


def test_ensure_onnx_weights_can_be_disabled(monkeypatch: Any) -> None:
    monkeypatch.setenv("DEPTHLENS_AUTO_EXPORT_ONNX", "false")

    result = benchmarks._ensure_onnx_weights("MiDaS_small", _missing_diag("cpu"))

    assert result["state"] == "missing_weights"
