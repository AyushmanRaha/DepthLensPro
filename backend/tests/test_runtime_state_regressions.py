from __future__ import annotations

from backend.services import runtime_state


def test_runtime_state_tracks_and_decrements_success_and_error():
    before = runtime_state.snapshot()["total_active_operations"]
    with runtime_state.track_operation("inference", "estimate:test"):
        snap = runtime_state.snapshot()
        assert snap["busy"] is True
        assert snap["active_operations"]["active_inference"] >= 1
        assert snap["last_operation"] == "estimate:test"
    assert runtime_state.snapshot()["total_active_operations"] == before

    try:
        with runtime_state.track_operation("detection", "detect:test"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert runtime_state.snapshot()["total_active_operations"] == before


def test_benchmark_compat_uses_generic_tracker():
    runtime_state.set_benchmark_busy(True)
    try:
        snap = runtime_state.snapshot()
        assert runtime_state.benchmark_busy() is True
        assert snap["active_operations"]["active_benchmark"] >= 1
    finally:
        runtime_state.set_benchmark_busy(False)
    assert runtime_state.benchmark_busy() is False
