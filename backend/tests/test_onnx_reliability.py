from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.scripts import export_onnx
from backend.services import benchmarks, inference, onnx_diagnostics


class FakeSession:
    def __init__(
        self, path: str, sess_options: Any = None, providers: list[str] | None = None
    ) -> None:
        self.path = path
        self.sess_options = sess_options
        self.providers = providers or []


def install_fake_ort(monkeypatch: pytest.MonkeyPatch, *, fail_load: bool = False) -> list[str]:
    loaded_paths: list[str] = []

    def inference_session(
        path: str, sess_options: Any = None, providers: list[str] | None = None
    ) -> FakeSession:
        loaded_paths.append(path)
        if fail_load:
            raise RuntimeError("bad protobuf")
        return FakeSession(path, sess_options=sess_options, providers=providers)

    fake = types.SimpleNamespace(
        __version__="test",
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=inference_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return loaded_paths


def test_corrupt_onnx_file_is_invalid_not_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "midas_small.onnx"
    path.write_bytes(b"not an onnx")
    monkeypatch.setenv("DEPTHLENS_ONNX_DIR", str(tmp_path))
    install_fake_ort(monkeypatch, fail_load=True)

    status = onnx_diagnostics.onnx_model_status("midas_small", "cpu")

    assert status["state"] == "invalid/corrupt"
    assert status["selected_path"] == str(path.resolve())
    assert status["error_code"] == "ONNX_SESSION_INIT_FAILED"


def test_create_onnx_session_uses_explicit_path_without_reresolving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "custom.onnx"
    explicit.write_bytes(b"onnx")
    loaded_paths = install_fake_ort(monkeypatch)
    monkeypatch.setattr(
        onnx_diagnostics,
        "resolve_onnx_path",
        lambda model: pytest.fail("explicit model_path should not be re-resolved"),
    )

    result = onnx_diagnostics.create_onnx_session("midas_small", "cpu", model_path=explicit)

    assert result["ok"] is True
    assert loaded_paths == [str(explicit.resolve())]


def test_atomic_export_preserves_final_file_after_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = tmp_path / "midas_small.onnx"
    final.write_bytes(b"good-existing")

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def cpu(self) -> "FakeModel":
            return self

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(export_onnx.torch.hub, "load", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(export_onnx.torch, "zeros", lambda *args, **kwargs: object())
    monkeypatch.setattr(export_onnx.torch, "no_grad", lambda: FakeNoGrad())

    def fake_export(model: Any, args: tuple[Any, ...], output_path: str, **kwargs: Any) -> None:
        Path(output_path).write_bytes(b"partial")

    monkeypatch.setattr(export_onnx.torch.onnx, "export", fake_export)
    monkeypatch.setattr(
        export_onnx, "_validate_onnx", lambda path, model: (False, "simulated failure")
    )

    result = export_onnx.export_model_to_onnx("midas_small", force=True, output_dir=tmp_path)

    assert result["ok"] is False
    assert final.read_bytes() == b"good-existing"
    assert not list(tmp_path.glob("*.tmp.onnx"))


def test_benchmark_auto_export_is_locked_for_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DEPTHLENS_AUTO_EXPORT_ONNX", "true")
    calls: list[str] = []
    status_calls = 0
    status_lock = threading.Lock()

    def fake_status(model: str, device: str = "auto") -> dict[str, Any]:
        nonlocal status_calls
        with status_lock:
            status_calls += 1
            state = "available" if calls else "missing"
        return {"state": state, "runtime": {"requested_device": device}, "path": {}}

    def fake_export(model: str, output_dir: Path) -> Path:
        calls.append(model)
        return output_dir / "midas_small.onnx"

    monkeypatch.setattr(benchmarks, "onnx_model_status", fake_status)
    monkeypatch.setattr(benchmarks, "onnx_model_path", lambda model: tmp_path / "midas_small.onnx")
    monkeypatch.setattr("backend.scripts.export_onnx.export_model", fake_export)

    diag = {"state": "missing", "runtime": {"requested_device": "cpu"}, "path": {}}
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(benchmarks._ensure_onnx_weights("midas_small", diag))
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == ["midas_small"]
    assert [result["state"] for result in results] == ["available", "available"]
    assert status_calls >= 2


def test_onnx_failure_metadata_is_visible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        inference,
        "resolve_onnx_path",
        lambda model: {
            "model_id": model,
            "onnx_path": str(tmp_path / "midas_small.onnx"),
            "expected_path": str(tmp_path / "midas_small.onnx"),
            "exists": True,
            "size_bytes": 4,
            "error": None,
        },
    )
    monkeypatch.setattr(
        inference,
        "_infer_onnx",
        lambda *args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        inference,
        "_infer_torch",
        lambda img, model, device: np.zeros(img.shape[:2], dtype=np.float32),
    )

    _depth, meta = inference._infer_with_metadata(
        np.zeros((2, 3, 3), dtype=np.uint8), "midas_small", "cpu", "onnx"
    )

    assert meta["engine_requested"] == "onnx"
    assert meta["engine_used"] == "pytorch"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"]
    assert meta["onnx_path"] == str(tmp_path / "midas_small.onnx")
    assert meta["onnx"]["runtime_error"] == "RuntimeError: boom"


def test_onnx_depth_resize_clamps_bicubic_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.depth_models import resize_onnx_depth

    def fake_resize(data: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        del interpolation
        return np.full((size[1], size[0]), 1.2, dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "cv2", types.SimpleNamespace(INTER_CUBIC=2, resize=fake_resize)
    )
    pred = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    resized = resize_onnx_depth(pred, (9, 9))

    assert resized.shape == (9, 9)
    assert resized.dtype == np.float32
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 1.0


def test_onnx_engine_load_is_singleton_and_forward_lock_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference.clear_models()
    created: list[str] = []
    create_lock = threading.Lock()

    class FakeEngine:
        provider = "CPUExecutionProvider"

        def __init__(self, model_name: str, device: str) -> None:
            with create_lock:
                created.append(f"{model_name}:{device}")
            self.model_name = model_name
            self.device = device

    monkeypatch.setattr(inference, "ONNXExecutionEngine", FakeEngine)
    engines: list[Any] = []
    threads = [
        threading.Thread(
            target=lambda: engines.append(inference._load_onnx_engine("MiDaS_small", "cpu"))
        )
        for _ in range(4)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert created == ["midas_small:cpu"]
    assert len({id(engine) for engine in engines}) == 1
    assert "midas_small:cpu" in inference._ONNX_FORWARD_LOCKS
    inference.clear_models()
