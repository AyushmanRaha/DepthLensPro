from backend.services import engine_selector as es

VERIFIED_ONNX = {
    "exists": True,
    "size_bytes": 123,
    "state": "available",
    "session_available": True,
    "runtime_importable": True,
    "provider_available": True,
    "providers_used": ["CPUExecutionProvider"],
    "providers_attempted": [["CPUExecutionProvider"]],
}
ASSET_ONLY = {"exists": True, "size_bytes": 123}


def setup_function():
    es.clear_engine_decisions()


def test_normalize_aliases():
    assert es.normalize_engine_mode(None) == "auto"
    assert es.normalize_engine_mode("onnx") == "onnxruntime"
    assert es.normalize_engine_mode("compare", allow_both=True) == "both"


def test_forced_and_defaults(monkeypatch):
    monkeypatch.setattr(es, "provider_signature", lambda d, m: "CPUExecutionProvider")
    monkeypatch.setattr(es, "onnx_file_fingerprint", lambda m: "hash1")

    assert (
        es.select_engine_for_inference("MiDaS_small", "cpu", "pytorch", VERIFIED_ONNX)[
            "selected_engine"
        ]
        == "pytorch"
    )
    assert (
        es.select_engine_for_inference("MiDaS_small", "cpu", "onnx", VERIFIED_ONNX)[
            "selected_engine"
        ]
        == "onnxruntime"
    )
    assert (
        es.select_engine_for_inference("MiDaS_small", "cpu", "auto", VERIFIED_ONNX)[
            "selected_engine"
        ]
        == "onnxruntime"
    )
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", VERIFIED_ONNX)[
            "selected_engine"
        ]
        == "pytorch"
    )
    asset_auto = es.select_engine_for_inference("MiDaS_small", "cpu", "auto", ASSET_ONLY)
    assert asset_auto["selected_engine"] == "pytorch"
    assert "session unverified" in asset_auto["reason"]
    asset_forced = es.select_engine_for_inference("MiDaS_small", "cpu", "onnx", ASSET_ONLY)
    assert asset_forced["selected_engine"] == "pytorch"
    assert asset_forced["source"] == "forced_fallback"
    assert asset_forced["fallback_target"] == "pytorch"


def test_cached_benchmark_margin_and_fingerprint(monkeypatch):
    fp = {"value": "hash1"}
    monkeypatch.setattr(es, "provider_signature", lambda d, m: "CPUExecutionProvider")
    monkeypatch.setattr(es, "onnx_file_fingerprint", lambda m: fp["value"])
    result = {
        "model_id": "DPT_Hybrid",
        "device_resolved": "cpu",
        "pytorch": {"latency_ms": 100.0},
        "onnx": {"latency_ms": 80.0},
    }
    es.record_benchmark_decision(result)
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", VERIFIED_ONNX)[
            "selected_engine"
        ]
        == "onnxruntime"
    )
    fp["value"] = "hash2"
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", VERIFIED_ONNX)[
            "selected_engine"
        ]
        == "pytorch"
    )
    fp["value"] = "hash1"
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", ASSET_ONLY)["selected_engine"]
        == "pytorch"
    )
