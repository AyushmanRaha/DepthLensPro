"""Final cleanup regression tests for inference architecture and diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app
from backend.services import cache_service, inference

client = TestClient(app)


def _patch_cv2_for_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def imdecode(arr: np.ndarray, flags: int) -> np.ndarray:
        del arr, flags
        return np.full((3, 4, 3), 64, dtype=np.uint8)

    def cvt_color(img: np.ndarray, code: int) -> np.ndarray:
        if code == inference.cv2.COLOR_BGR2GRAY:
            return np.full(img.shape[:2], 128, dtype=np.uint8)
        if code == inference.cv2.COLOR_GRAY2BGR:
            return np.repeat(img[:, :, None], 3, axis=2)
        return img[..., ::-1] if img.ndim == 3 else img

    def apply_color_map(img: np.ndarray, cmap: int) -> np.ndarray:
        del cmap
        return np.repeat(img[:, :, None], 3, axis=2)

    def imencode(ext: str, img: np.ndarray) -> tuple[bool, np.ndarray]:
        del ext, img
        return True, np.frombuffer(b"png", dtype=np.uint8)

    monkeypatch.setattr(inference.cv2, "imdecode", imdecode)
    monkeypatch.setattr(inference.cv2, "cvtColor", cvt_color)
    monkeypatch.setattr(inference.cv2, "applyColorMap", apply_color_map)
    monkeypatch.setattr(inference.cv2, "imencode", imencode)


def test_depth_estimator_is_legacy_and_not_active_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.depth_models import DepthEstimator

    assert getattr(DepthEstimator, "__legacy__") is True

    def fail_init(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("active inference must not instantiate DepthEstimator")

    monkeypatch.setattr(DepthEstimator, "__init__", fail_init)
    monkeypatch.setattr(
        inference,
        "_infer_torch",
        lambda img, model, device: np.zeros(img.shape[:2], dtype=np.float32),
    )
    monkeypatch.setattr(inference, "resolve_onnx_path", lambda model: {"exists": False})

    depth, meta = inference._infer_with_metadata(
        np.zeros((2, 2, 3), dtype=np.uint8), "MiDaS_small", "cpu"
    )

    assert depth.shape == (2, 2)
    assert meta["engine_used"] == "pytorch"


def test_model_metadata_is_lightweight_registry_alias() -> None:
    before_modules = set(sys.modules)

    import backend.model_metadata as model_metadata
    from backend.model_registry import supported_models_legacy_payload

    newly_imported = set(sys.modules) - before_modules
    assert model_metadata.SUPPORTED_MODELS == supported_models_legacy_payload()
    assert "torch" not in newly_imported
    assert "onnxruntime" not in newly_imported


def test_active_inference_path_with_deterministic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    inference.clear_models()
    cache_service.clear()
    _patch_cv2_for_inference(monkeypatch)

    calls = {"infer": 0}

    def fake_infer(img: np.ndarray, model: str, device: str) -> tuple[np.ndarray, dict[str, Any]]:
        calls["infer"] += 1
        assert model == "midas_small"
        assert device == "cpu"
        depth = np.linspace(0.0, 1.0, img.shape[0] * img.shape[1], dtype=np.float32).reshape(
            img.shape[:2]
        )
        return depth, {
            "model_id": model,
            "model_display_name": "MiDaS Small",
            "engine_requested": "auto",
            "engine_used": "pytorch",
            "device_requested": device,
            "device_used": device,
            "fallback_used": False,
            "warnings": [],
        }

    monkeypatch.setattr(inference, "_infer_with_metadata", fake_infer)

    payload = inference.process_image(
        b"image", "MiDaS Small", "inferno", "cpu", "sample.png", metrics="fast"
    )

    assert calls["infer"] == 1
    assert payload["model_id"] == "midas_small"
    assert payload["engine_used"] == "pytorch"
    assert payload["resolution"] == {"width": 4, "height": 3}
    assert payload["depth_map"]
    assert payload["metrics"]["histogram"]["counts"]


def test_gt_mode_reuses_depth_cache_without_full_response_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference.clear_models()
    _patch_cv2_for_inference(monkeypatch)
    calls = {"infer": 0}

    def fake_infer(img: np.ndarray, model: str, device: str) -> tuple[np.ndarray, dict[str, Any]]:
        calls["infer"] += 1
        return np.full(img.shape[:2], 0.5, dtype=np.float32), {
            "model_id": model,
            "model_display_name": "MiDaS Small",
            "engine_requested": "auto",
            "engine_used": "pytorch",
            "device_requested": device,
            "device_used": device,
            "fallback_used": False,
            "warnings": [],
        }

    monkeypatch.setattr(inference, "_infer_with_metadata", fake_infer)
    monkeypatch.setattr(
        inference,
        "decode_ground_truth",
        lambda raw, filename, invalid_value=None, scale=None: SimpleNamespace(
            depth=np.full((3, 4), 0.5, dtype=np.float32), metadata={"filename": filename}
        ),
    )
    monkeypatch.setattr(
        inference,
        "compute_ground_truth_metrics",
        lambda depth, gt, metadata, invalid_value=None: {
            "metrics": {"gt_mae": float(np.abs(depth - gt).mean())},
            "metadata": {**metadata, "provided": True},
            "visualizations": {},
            "warnings": [],
            "unavailable": {},
        },
    )

    first = inference.process_image(
        b"same-image", "midas_small", "inferno", "cpu", "a.png", gt_raw=b"gt-1"
    )
    second = inference.process_image(
        b"same-image", "midas_small", "inferno", "cpu", "a.png", gt_raw=b"gt-2"
    )

    assert calls["infer"] == 1
    assert first["cached"] is False and second["cached"] is False
    assert first["depth_cached"] is False and second["depth_cached"] is True
    assert second["metrics"]["gt_metrics"]["gt_mae"] == 0.0


def test_onnx_missing_fallback_metadata_includes_expected_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = tmp_path / "midas_small.onnx"
    monkeypatch.setattr(
        inference,
        "resolve_onnx_path",
        lambda model: {
            "model_id": model,
            "onnx_path": None,
            "expected_path": str(expected),
            "exists": False,
            "size_bytes": None,
            "error": "missing_file",
        },
    )
    monkeypatch.setattr(
        inference,
        "_infer_torch",
        lambda img, model, device: np.zeros(img.shape[:2], dtype=np.float32),
    )

    _depth, meta = inference._infer_with_metadata(
        np.zeros((2, 3, 3), dtype=np.uint8), "midas_small", "cpu"
    )

    assert meta["engine_used"] == "pytorch"
    assert meta["fallback_used"] is True
    assert meta["onnx_path"] == str(expected)
    assert meta["onnx"]["error"] == "missing_file"


def test_model_and_device_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes,
        "_cached_devices",
        lambda force=False: (
            {"cpu": {"available": True}, "cuda:0": {"available": True}},
            "cuda:0",
            {"cached": False},
        ),
    )
    monkeypatch.setattr(
        routes, "_resolve", lambda requested: "cuda:0" if requested == "auto" else requested
    )

    assert routes._validated_device_or_422("auto") == "cuda:0"
    assert routes._validated_device_or_422("cpu") == "cpu"
    assert inference.normalize_model_id("DPT-Hybrid") == "dpt_hybrid"


def test_settings_env_dotenv_booleans_ints_and_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from backend import config

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "PORT=9001\nDEBUG=true\nCACHE_MAX_ENTRIES=42\nDEPTHLENS_MAX_DIM=1024\nLOG_LEVEL=debug\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PORT", "9002")
    monkeypatch.setenv("DEPTHLENS_PRELOAD_MODEL", "1")
    monkeypatch.delenv("REDIS_PORT", raising=False)

    values = config._settings_values()
    settings = config.Settings(**values)

    assert settings.PORT == 9002
    assert settings.DEBUG is True
    assert settings.CACHE_MAX_ENTRIES == 42
    assert settings.DEPTHLENS_PRELOAD_MODEL is True
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.REDIS_PORT == 6379
    assert settings.DEPTHLENS_DEFAULT_METRICS == "fast"


def test_health_caches_acceleration_and_readiness_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes,
        "_DEVICE_CACHE",
        {"expires_at": 0.0, "devices": None, "primary": "cpu", "error": None},
    )
    monkeypatch.setattr(
        routes,
        "_ACCEL_CACHE",
        {"expires_at": 0.0, "checks": None, "error": None, "device_keys": ()},
    )
    monkeypatch.setattr(
        routes,
        "_READINESS_CACHE",
        {"expires_at": 0.0, "device": None, "payload": None, "error": None},
    )
    monkeypatch.setattr(
        routes,
        "_available_devices",
        lambda: {"cpu": {"available": True, "hardware_name": "test", "name": "CPU · test"}},
    )
    monkeypatch.setattr(routes, "_default_device_key", lambda: "cpu")
    calls = {"accel": 0, "ready": 0}

    def fake_accel(devs: dict[str, Any]) -> dict[str, Any]:
        calls["accel"] += 1
        return {"cuda": {"available": False, "operational": False}}

    def fake_ready(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["ready"] += 1
        return {"overall_status": "pytorch_ready_onnx_unavailable", "models": {}}, {"cached": False}

    monkeypatch.setattr(routes, "_acceleration_checks", fake_accel)
    monkeypatch.setattr(routes, "_cached_readiness_payload", fake_ready)
    monkeypatch.setattr(routes, "onnx_status_payload", lambda device: {"status": "ok"})

    assert client.get("/live").status_code == 200
    first = client.get("/health")
    second = client.get("/health")

    assert first.status_code == 200 and second.status_code == 200
    assert calls["accel"] == 1
    assert calls["ready"] == 2  # route calls the cache wrapper; internals are tested separately


def test_readiness_cache_wrapper_reuses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes,
        "_READINESS_CACHE",
        {"expires_at": 0.0, "device": None, "payload": None, "error": None},
    )
    calls = {"count": 0}

    def fake_payload(device: str) -> dict[str, Any]:
        calls["count"] += 1
        return {"overall_status": f"ok:{device}", "models": {}}

    monkeypatch.setattr(routes, "readiness_payload", fake_payload)

    first, first_meta = routes._cached_readiness_payload("cpu")
    second, second_meta = routes._cached_readiness_payload("cpu")

    assert first == second == {"overall_status": "ok:cpu", "models": {}}
    assert first_meta["cached"] is False
    assert second_meta["cached"] is True
    assert calls["count"] == 1


def test_full_metrics_histogram_entropy_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sobel(src: np.ndarray, ddepth: int, dx: int, dy: int, ksize: int) -> np.ndarray:
        del ddepth, ksize
        return np.gradient(src.astype(np.float64), axis=1 if dx else 0 if dy else 0)

    monkeypatch.setattr(inference.cv2, "Sobel", fake_sobel)
    monkeypatch.setattr(
        inference.cv2,
        "cvtColor",
        lambda img, code: np.full(img.shape[:2], 128, dtype=np.uint8),
    )
    monkeypatch.setattr(inference.cv2, "GaussianBlur", lambda x, ksize, sigma: x)

    depth = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    metrics = inference._compute_metrics(depth, img)
    expected_hist, expected_edges = np.histogram(depth.ravel(), bins=32, range=(0.0, 1.0))
    expected_256, _ = np.histogram(depth.ravel(), bins=256, range=(0.0, 1.0))
    hp = expected_256 / expected_256.sum()
    expected_entropy = float(-np.sum(hp[hp > 0] * np.log2(hp[hp > 0])))

    assert metrics["histogram"]["counts"] == expected_hist.tolist()
    assert metrics["histogram"]["bin_edges"] == [round(float(e), 3) for e in expected_edges]
    assert metrics["entropy"] == round(expected_entropy, 3)
