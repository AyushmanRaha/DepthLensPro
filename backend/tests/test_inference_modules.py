from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np

from backend.services import image_io, inference, inference_cache, metrics


def _png_bytes(width: int = 8, height: int = 6) -> bytes:
    return f"fake-png:{width}:{height}".encode()


def test_image_io_decode_resize_normalize_colorize_and_base64(monkeypatch: Any) -> None:
    raw = _png_bytes(width=320, height=160)
    source = np.zeros((160, 320, 3), dtype=np.uint8)

    monkeypatch.setattr(image_io.cv2, "imdecode", lambda arr, flag: source.copy())
    monkeypatch.setattr(
        image_io.cv2,
        "resize",
        lambda img, size, interpolation: np.zeros((size[1], size[0], 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        image_io.cv2,
        "applyColorMap",
        lambda u8, cmap: np.dstack([u8, u8, u8]).astype(np.uint8),
    )
    monkeypatch.setattr(
        image_io.cv2, "imencode", lambda ext, img: (True, np.frombuffer(b"encoded", dtype=np.uint8))
    )

    decoded = image_io._decode(raw, max_dim=256)
    depth = image_io._normalize_depth(np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32))
    color = image_io._colorize(depth, "inferno")
    encoded = image_io._b64(color)

    assert decoded.shape[:2] == (128, 256)
    assert float(depth.min()) == 0.0
    assert float(depth.max()) > 0.999
    assert color.shape == (2, 2, 3)
    assert base64.b64decode(encoded) == b"encoded"


def test_inference_cache_keys_and_cached_depth_are_stable_copies() -> None:
    raw = b"same-image"
    depth = np.arange(4, dtype=np.float32).reshape(2, 2)
    resolution = {"width": 2, "height": 2}
    key = inference_cache._depth_cache_key(raw, "midas_small", "cpu", 512)

    inference_cache._set_cached_depth(key, depth, resolution)
    cached = inference_cache._get_cached_depth(key)
    assert cached is not None
    cached_depth, cached_resolution = cached
    cached_depth[0, 0] = 999
    cached_resolution["width"] = 999

    cached_again = inference_cache._get_cached_depth(key)
    assert cached_again is not None
    assert cached_again[0][0, 0] == 0
    assert cached_again[1] == resolution
    assert inference_cache._raw_hash(raw) == inference._raw_hash(raw)
    assert inference_cache._fhash(raw, "midas_small", "inferno", "cpu") == inference._fhash(
        raw, "midas_small", "inferno", "cpu"
    )


def test_metrics_parsing_and_grouped_payload_modes() -> None:
    assert metrics.normalize_metrics_mode(" FAST ") == "fast"
    assert metrics.parse_outputs("depth,grayscale,color") == ("color", "gray")

    depth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    img = np.dstack([np.full((4, 4), 127, dtype=np.uint8)] * 3)
    fast_payload = metrics._metrics_for_mode(depth, img, "fast")
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(metrics.cv2, "Sobel", lambda src, ddepth, dx, dy, ksize: np.ones_like(src))
    monkeypatch.setattr(metrics.cv2, "cvtColor", lambda image, code: image[:, :, 0])
    monkeypatch.setattr(metrics.cv2, "GaussianBlur", lambda src, ksize, sigma: src)
    try:
        full_payload = metrics._metrics_for_mode(depth, img, "full")
    finally:
        monkeypatch.undo()

    assert "histogram" in fast_payload["prediction_stats"]
    assert fast_payload["unavailable"]["rmse"] == "not_requested_fast_mode"
    assert "rmse" in full_payload["proxy_metrics"]
    assert full_payload["gt_metrics"] == {}


def test_onnx_missing_falls_back_to_pytorch_with_same_metadata(
    monkeypatch: Any, tmp_path: Path
) -> None:
    expected = tmp_path / "midas_small.onnx"
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    fallback_depth = np.full((3, 4), 0.5, dtype=np.float32)

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
    monkeypatch.setattr(inference, "_infer_torch", lambda img, model, device: fallback_depth)

    depth, metadata = inference._infer_with_metadata(image, "midas_small", "cpu", "onnx")

    assert np.array_equal(depth, fallback_depth)
    assert metadata["engine_requested"] == "onnxruntime"
    assert metadata["engine_used"] == "pytorch"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"]
    assert metadata["onnx_path"] == str(expected)
    assert metadata["onnx"]["expected_path"] == str(expected)


def test_process_image_top_level_contract_keys(monkeypatch: Any) -> None:
    raw = _png_bytes()
    monkeypatch.setattr(
        inference, "_decode", lambda data, max_dim=None: np.zeros((6, 8, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(inference, "_b64", lambda img: "encoded")
    monkeypatch.setattr(
        inference, "_colorize", lambda depth, cmap: np.zeros((6, 8, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        inference.cv2, "cvtColor", lambda image, code: np.zeros((6, 8, 3), dtype=np.uint8)
    )

    expected_keys = {
        "metrics",
        "latency_ms",
        "model",
        "model_id",
        "model_display_name",
        "colormap",
        "device_used",
        "resolution",
        "filename",
        "cached",
        "depth_cached",
        "metrics_mode",
        "outputs",
        "gt_metadata",
        "depth_map",
        "grayscale",
        "engine_requested",
        "engine_used",
        "device_requested",
        "fallback_used",
        "warnings",
        "onnx",
    }

    monkeypatch.setattr(
        inference,
        "_infer_with_metadata",
        lambda img, model, device: (
            np.zeros(img.shape[:2], dtype=np.float32),
            {
                "model_id": "midas_small",
                "model_display_name": "MiDaS Small",
                "engine_requested": "auto",
                "engine_used": "pytorch",
                "device_requested": device,
                "device_used": device,
                "fallback_used": False,
                "warnings": [],
                "onnx": None,
            },
        ),
    )
    inference.clear_models()

    payload = inference.process_image(
        raw,
        "midas_small",
        "inferno",
        "cpu",
        "sample.png",
        metrics="fast",
        outputs="color,gray",
    )

    assert expected_keys <= set(payload)
