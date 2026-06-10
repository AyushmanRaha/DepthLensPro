from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from backend.services.onnx_diagnostics import create_onnx_session


class FakeOrt:
    __version__ = "test"

    class SessionOptions:
        pass

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    @staticmethod
    def InferenceSession(
        *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - should not be called in missing tests
        raise AssertionError("InferenceSession should not be called for invalid paths")


def test_inference_session_not_called_when_file_missing(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    missing = tmp_path / "midas_small.onnx"
    result = create_onnx_session("midas_small", "cpu", model_path=missing)
    assert result["ok"] is False
    assert result["error_code"] == "ONNX_MODEL_MISSING"
    assert result["fallback_allowed"] is True


def test_provider_selection_uses_available_providers(monkeypatch: Any, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    class GoodOrt(FakeOrt):
        @staticmethod
        def InferenceSession(
            path: str, sess_options: Any = None, providers: list[str] | None = None
        ) -> Any:
            calls["path"] = path
            calls["providers"] = providers
            return SimpleNamespace(get_inputs=lambda: [], get_outputs=lambda: [])

    model = tmp_path / "midas_small.onnx"
    model.write_bytes(b"not really onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime", GoodOrt)
    result = create_onnx_session("MiDaS_small", "cuda", model_path=model)
    assert result["ok"] is True
    assert calls["providers"] == ["CPUExecutionProvider"]
    assert calls["path"] == str(model)
