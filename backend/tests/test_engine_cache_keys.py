from backend.services.inference_cache import _depth_cache_key, _fhash


def test_engine_separates_response_and_depth_cache_keys():
    raw = b"image"
    assert _fhash(raw, "MiDaS_small", "inferno", "cpu", engine="pytorch") != _fhash(
        raw, "MiDaS_small", "inferno", "cpu", engine="onnxruntime"
    )
    assert _depth_cache_key(raw, "MiDaS_small", "cpu", None, "pytorch") != _depth_cache_key(
        raw, "MiDaS_small", "cpu", None, "onnxruntime"
    )
