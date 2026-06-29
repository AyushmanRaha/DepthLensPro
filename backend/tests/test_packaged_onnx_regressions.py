from __future__ import annotations

import asyncio
import importlib
import sys
import types


def test_live_import_is_lightweight():
    blocked = {
        "backend.services.benchmarks",
        "backend.services.inference",
        "backend.depth_models",
        "torch",
        "cv2",
        "numpy",
        "onnxruntime",
    }
    for name in list(blocked):
        sys.modules.pop(name, None)

    class FakeRouter:
        def get(self, _path):
            return lambda func: func

    sys.modules.setdefault("fastapi", types.SimpleNamespace(APIRouter=lambda: FakeRouter()))
    live = importlib.import_module("backend.api.live")
    asyncio.run(live.live())
    assert blocked.isdisjoint(sys.modules)


def test_onnx_healthy_resolves_requested_model(monkeypatch):
    from backend.services import engine_selector

    seen = []

    def fake_resolve(model_id: str):
        seen.append(model_id)
        return {"exists": True, "size_bytes": 10}

    monkeypatch.setattr(engine_selector, "resolve_onnx_path", fake_resolve)
    assert engine_selector._onnx_healthy(None, "dpt_hybrid") is True
    assert seen == ["dpt_hybrid"]


def test_create_onnx_session_retries_cpu_provider(monkeypatch, tmp_path):
    from backend.services import onnx_diagnostics

    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    calls = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            calls.append(providers)
            if providers != ["CPUExecutionProvider"]:
                raise RuntimeError("preferred failed")

        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    result = onnx_diagnostics.create_onnx_session("midas_small", "cuda", model_path=model)
    assert result["ok"] is True
    assert result["providers_used"] == ["CPUExecutionProvider"]
    assert result["provider_fallback_used"] is True
    assert len(result["provider_failure_chain"]) >= 1


def _import_object_detection_without_pil():
    image_mod = types.SimpleNamespace()
    sys.modules.setdefault(
        "PIL", types.SimpleNamespace(Image=image_mod, UnidentifiedImageError=OSError)
    )
    sys.modules.setdefault("PIL.Image", image_mod)
    from backend.services import object_detection

    return object_detection


def test_detector_status_does_not_warm_without_request(monkeypatch):
    object_detection = _import_object_detection_without_pil()

    def fail_get_detector(*args, **kwargs):
        raise AssertionError("status warmup=false must not load detector")

    monkeypatch.setattr(object_detection, "get_detector", fail_get_detector)
    payload = object_detection.detector_status("cpu", warmup=False)
    assert payload["state"] in {"idle", "error", "missing_weights", "dependency_missing"}
    assert payload["warmup_in_progress"] is False


def test_detector_status_reports_missing_dependency(monkeypatch):
    object_detection = _import_object_detection_without_pil()

    def fail_get_detector(*args, **kwargs):
        err = object_detection.DetectorUnavailableError("missing deps")
        err.error_code = "DETECTOR_DEPENDENCY_MISSING"
        raise err

    monkeypatch.setattr(object_detection, "get_detector", fail_get_detector)
    payload = object_detection.detector_status("cpu", warmup=True)
    assert payload["state"] == "dependency_missing"
    assert payload["available"] is False
