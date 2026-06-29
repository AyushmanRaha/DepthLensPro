from backend.services import engine_selector as es


def setup_function():
    es.clear_engine_decisions()


def test_normalize_aliases():
    assert es.normalize_engine_mode(None) == "auto"
    assert es.normalize_engine_mode("onnx") == "onnxruntime"
    assert es.normalize_engine_mode("compare", allow_both=True) == "both"


def test_forced_and_defaults(monkeypatch):
    monkeypatch.setattr(es, "provider_signature", lambda d, m: "CPUExecutionProvider")
    monkeypatch.setattr(es, "onnx_file_fingerprint", lambda m: "hash1")
    healthy = {"exists": True, "size_bytes": 123}
    assert (
        es.select_engine_for_inference("MiDaS_small", "cpu", "pytorch", healthy)["selected_engine"]
        == "pytorch"
    )
    assert (
        es.select_engine_for_inference("MiDaS_small", "cpu", "onnx", healthy)["selected_engine"]
        == "onnxruntime"
    )
    assert (
        es.select_engine_for_inference("MiDaS_small", "cpu", "auto", healthy)["selected_engine"]
        == "onnxruntime"
    )
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", healthy)["selected_engine"]
        == "pytorch"
    )


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
    healthy = {"exists": True, "size_bytes": 123}
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", healthy)["selected_engine"]
        == "onnxruntime"
    )
    fp["value"] = "hash2"
    assert (
        es.select_engine_for_inference("DPT_Hybrid", "cpu", "auto", healthy)["selected_engine"]
        == "pytorch"
    )
