from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.services import benchmarks, inference


def test_missing_onnx_uses_pytorch_fallback(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        inference,
        "resolve_onnx_path",
        lambda model: {
            "model_id": model,
            "onnx_path": str(tmp_path / "missing.onnx"),
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
    depth, meta = inference._infer_with_metadata(
        np.zeros((4, 4, 3), dtype=np.uint8), "MiDaS Small", "cpu", "onnx"
    )
    assert depth.shape == (4, 4)
    assert meta["engine_requested"] == "onnxruntime"
    assert meta["engine_used"] == "pytorch"
    assert meta["fallback_used"] is True
    assert meta["model_id"] == "midas_small"


def test_benchmark_failed_onnx_has_null_throughput_and_speedup(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmarks, "_resolve", lambda device: "cpu")
    monkeypatch.setattr(
        benchmarks,
        "_infer_torch",
        lambda frame, model, device: np.zeros(frame.shape[:2], dtype=np.float32),
    )
    monkeypatch.setattr(benchmarks, "onnx_model_path", lambda model: tmp_path / f"{model}.onnx")
    monkeypatch.setattr(
        benchmarks,
        "onnx_model_status",
        lambda model, device="auto": {
            "state": "export_failed",
            "export_error": "SymInt",
            "runtime": {"requested_device": device},
            "recommended_export_command": None,
        },
    )
    result = benchmarks.run_benchmark("DPT Hybrid", "cpu", 1)
    onnx = next(r for r in result["results"] if r["engine"] == "onnxruntime")
    assert onnx["throughput_fps"] is None
    assert result["speedup"] is None
    assert result["pytorch"]["latency_ms"] is not None
    assert result["onnx"]["status"] == "export_failed"
