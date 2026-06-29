"""Benchmark helpers for comparing PyTorch and ONNX Runtime inference paths."""

from __future__ import annotations

import logging
import os
import platform
import resource
import threading
import time
from statistics import mean
from typing import Any, Callable, cast

import numpy as np

from backend.depth_models import onnx_model_path
from backend.model_registry import get_model_spec, normalize_model_id
from backend.services import observability
from backend.services.engine_selector import (
    ENGINE_SELECTION_MARGIN,
    record_benchmark_decision,
    select_engine_for_inference,
)
from backend.services.inference import _infer_onnx, _infer_torch
from backend.services.onnx_diagnostics import onnx_model_status
from backend.services.runtime_state import benchmark_busy as _runtime_benchmark_busy
from backend.services.runtime_state import set_benchmark_busy
from backend.utils.hardware import _resolve

DEFAULT_BENCHMARK_ITERATIONS = 3
BENCHMARK_TIMEOUT_SECONDS = int(os.getenv("DEPTHLENS_BENCHMARK_TIMEOUT_SECONDS", "180"))
log = logging.getLogger("depthlens")


def benchmark_busy() -> bool:
    return _runtime_benchmark_busy()


_AUTO_EXPORT_LOCKS: dict[str, threading.Lock] = {}
_AUTO_EXPORT_LOCKS_GUARD = threading.Lock()


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a permissive boolean environment flag value."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _auto_export_lock(model: str) -> threading.Lock:
    """Return a per-model lock so concurrent benchmarks do not export the same graph."""

    with _AUTO_EXPORT_LOCKS_GUARD:
        return _AUTO_EXPORT_LOCKS.setdefault(model, threading.Lock())


def _ensure_onnx_weights(model: str, onnx_diag: dict[str, Any]) -> dict[str, Any]:
    """Export missing ONNX weights on demand for benchmark comparisons."""

    exportable_states = {
        "missing_weights",
        "missing_model_file",
        "missing",
        "invalid_checker",
        "invalid_session",
        "invalid_dummy_inference",
    }
    if onnx_diag.get("state") not in exportable_states:
        return onnx_diag
    if not _env_flag("DEPTHLENS_AUTO_EXPORT_ONNX", False):
        return {**onnx_diag, "auto_export_enabled": False}

    requested_device = str(onnx_diag.get("runtime", {}).get("requested_device", "auto"))
    with _auto_export_lock(model):
        refreshed = onnx_model_status(model, requested_device)
        if refreshed.get("state") not in exportable_states:
            return refreshed

        if refreshed.get("state") in {
            "invalid_checker",
            "invalid_session",
            "invalid_dummy_inference",
        }:
            corrupt_path = refreshed.get("selected_path") or refreshed.get("path", {}).get(
                "onnx_path"
            )
            if corrupt_path:
                try:
                    os.replace(corrupt_path, f"{corrupt_path}.corrupt")
                except OSError:
                    try:
                        os.remove(corrupt_path)
                    except OSError:
                        pass

        try:
            from backend.scripts.export_onnx import export_model

            output_dir = onnx_model_path(model).parent
            export_model(model, output_dir)
            return onnx_model_status(model, requested_device)
        except Exception as exc:
            failed = dict(refreshed)
            failed.update(
                {
                    "state": "export_failed",
                    "export_error": f"{type(exc).__name__}: {exc}",
                    "auto_export_enabled": True,
                }
            )
            return failed


def _memory_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 / 1024 if platform.system() != "Darwin" else 1 / (1024 * 1024)
    return round(usage * multiplier, 2)


def _memory_snapshot() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
            for line in meminfo_file:
                key, raw_value = line.split(":", 1)
                meminfo[key] = int(raw_value.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        meminfo = {}

    snapshot: dict[str, Any] = {"process_rss_mb": _memory_rss_mb()}
    if meminfo.get("MemTotal") and meminfo.get("MemAvailable") is not None:
        total = meminfo["MemTotal"]
        available = meminfo["MemAvailable"]
        snapshot.update(
            {
                "system_total_mb": round(total / 1024**2, 2),
                "system_available_mb": round(available / 1024**2, 2),
                "system_used_percent": round(((total - available) / total) * 100, 2),
            }
        )
    return snapshot


def _synthetic_frame(size: int = 384) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.uint8)
    return np.dstack((x, y, ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)))


def _measure(label: str, runner: Callable[[], np.ndarray], iterations: int) -> dict[str, Any]:
    latencies: list[float] = []
    start_memory = _memory_snapshot()
    t0 = time.perf_counter()
    runner()
    first_run_ms = (time.perf_counter() - t0) * 1000
    warmup_iterations = 1 if iterations > 1 else 0
    for _ in range(warmup_iterations):
        runner()
    for _ in range(iterations):
        t0 = time.perf_counter()
        runner()
        latencies.append((time.perf_counter() - t0) * 1000)
    end_memory = _memory_snapshot()
    avg = mean(latencies)
    return {
        "engine": label,
        "status": "ok",
        "state": "available",
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "first_run_ms": round(first_run_ms, 2),
        "cold_start_ms": round(first_run_ms, 2),
        "steady_state_latency_ms": round(avg, 2),
        "steady_state_throughput_fps": round(1000 / avg, 2) if avg else 0.0,
        "timing_mode": "warm_steady_state",
        "measurement_notes": (
            "First call recorded separately; warmup runs excluded from steady samples."
        ),
        "latency_ms": {
            "avg": round(avg, 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "samples": [round(v, 2) for v in latencies],
        },
        "throughput_fps": round(1000 / avg, 2) if avg else 0.0,
        "throughput_frames_per_min": round(60000 / avg, 2) if avg else 0.0,
        "memory": {
            "before": start_memory,
            "after": end_memory,
            "process_rss_delta_mb": round(
                end_memory["process_rss_mb"] - start_memory["process_rss_mb"], 2
            ),
        },
    }


def _unavailable(
    label: str, reason: str, state: str = "unavailable", extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = {
        "engine": label,
        "status": "unavailable",
        "state": state,
        "reason": reason,
        "latency_ms": None,
        "throughput_fps": None,
        "throughput_frames_per_min": None,
        "memory": None,
    }
    if extra:
        payload.update(extra)
    return payload


def run_benchmark(
    model: str = "MiDaS_small",
    device: str = "auto",
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    engine: str = "both",
) -> dict[str, Any]:
    """Return PyTorch-vs-ONNX latency, throughput, and memory benchmark matrices."""

    model = normalize_model_id(model)
    spec = get_model_spec(model)
    set_benchmark_busy(True)
    log.info("BENCHMARK_START model=%s device=%s iterations=%s", model, device, iterations)
    try:
        iterations = max(1, min(iterations, 20))
        engine = (engine or "both").strip().lower()
        if engine == "onnx":
            engine = "onnxruntime"
        if engine == "compare":
            engine = "both"
        if engine not in {"both", "auto", "pytorch", "onnxruntime"}:
            raise ValueError("engine must be one of: auto, both, onnxruntime, pytorch")
        resolved = str(_resolve(device))
        frame = _synthetic_frame()

        if engine == "onnxruntime":
            pytorch_result = _unavailable(
                "pytorch", "PyTorch benchmark not requested", "not_requested"
            )
        else:
            try:
                pytorch_result = _measure(
                    "pytorch", lambda: _infer_torch(frame, model, resolved), iterations
                )
            except Exception as exc:
                pytorch_result = _unavailable("pytorch", str(exc))

        weights_path = onnx_model_path(model)
        onnx_diag = _ensure_onnx_weights(model, onnx_model_status(model, resolved))
        state = onnx_diag.get("state")
        if engine == "pytorch":
            onnx_result = _unavailable(
                "onnxruntime", "ONNX Runtime benchmark not requested", "not_requested"
            )
        elif state == "available":
            try:
                onnx_result = _measure(
                    "onnxruntime", lambda: _infer_onnx(frame, model, resolved), iterations
                )
                onnx_result["provider"] = onnx_diag.get("runtime", {}).get("selected_provider")
                onnx_result["providers"] = onnx_diag.get("runtime", {}).get(
                    "selected_providers", []
                )
                onnx_result["uses_cpu_fallback"] = onnx_diag.get("runtime", {}).get(
                    "uses_cpu_fallback", False
                )
            except Exception as exc:
                onnx_result = _unavailable(
                    "onnxruntime", str(exc), "runtime_error", {"diagnostics": onnx_diag}
                )
        elif state in {"missing_weights", "missing_model_file", "missing", "optional_unavailable"}:
            command = onnx_diag.get("recommended_export_command")
            onnx_result = _unavailable(
                "onnxruntime",
                f"ONNX weights missing. Expected {weights_path}. Run {command}.",
                "missing_model_file",
                {"diagnostics": onnx_diag},
            )
        elif state in {"onnxruntime_missing", "runtime_unavailable"}:
            onnx_result = _unavailable(
                "onnxruntime",
                "onnxruntime is not installed or cannot be imported",
                "runtime_unavailable",
                {"diagnostics": onnx_diag},
            )
        elif state == "provider_unavailable":
            onnx_result = _unavailable(
                "onnxruntime",
                "No compatible ONNX Runtime execution provider is available",
                "provider_unavailable",
                {"diagnostics": onnx_diag},
            )
        elif state in {"invalid_checker", "invalid_session", "invalid_dummy_inference", "empty"}:
            unavailable_detail = (
                onnx_diag.get("technical_detail") or onnx_diag.get("message") or "validation failed"
            )
            onnx_result = _unavailable(
                "onnxruntime",
                f"ONNX unavailable ({state}): {unavailable_detail}",
                str(state),
                {"diagnostics": onnx_diag},
            )
        elif state == "export_failed":
            onnx_result = _unavailable(
                "onnxruntime",
                f"Automatic ONNX export failed: {onnx_diag.get('export_error', 'unknown error')}",
                "export_failed",
                {"diagnostics": onnx_diag},
            )
        else:
            onnx_result = _unavailable(
                "onnxruntime",
                "ONNX Runtime is unavailable",
                "unavailable",
                {"diagnostics": onnx_diag},
            )

        comparison: dict[str, Any] = {}
        pt_avg = (pytorch_result.get("latency_ms") or {}).get("avg")
        onnx_avg = (onnx_result.get("latency_ms") or {}).get("avg")
        provider = onnx_result.get("provider") or onnx_diag.get("runtime", {}).get(
            "selected_provider"
        )
        if (
            isinstance(pt_avg, (int, float))
            and isinstance(onnx_avg, (int, float))
            and onnx_avg > 0
            and pt_avg > 0
        ):
            speedup = round(pt_avg / onnx_avg, 2)
            onnx_factor = round(pt_avg / onnx_avg, 2)
            pytorch_factor = round(onnx_avg / pt_avg, 2)
            faster = "onnxruntime" if onnx_avg < pt_avg else "pytorch"
            recommended = (
                "onnxruntime" if onnx_avg <= pt_avg * ENGINE_SELECTION_MARGIN else "pytorch"
            )
            factor = onnx_factor if faster == "onnxruntime" else pytorch_factor
            display = (
                f"{'ONNX Runtime' if faster == 'onnxruntime' else 'PyTorch'} {factor:.1f}× faster"
            )
            reason = (
                ("ONNX Runtime faster by %.1f×" % onnx_factor)
                if recommended == "onnxruntime"
                else "PyTorch faster on this model/device"
            )
            comparison = {
                "latency_delta_ms": round(pt_avg - onnx_avg, 2),
                "speedup": speedup,
                "faster_engine": faster,
                "recommended_engine": recommended,
                "recommendation_reason": reason,
                "recommendation_source": "benchmark",
                "display_label": display,
                "onnx_faster_factor": onnx_factor if onnx_avg < pt_avg else None,
                "pytorch_faster_factor": pytorch_factor if pt_avg <= onnx_avg else None,
                "selection_margin": ENGINE_SELECTION_MARGIN,
                "uses_cpu_fallback": bool(onnx_result.get("uses_cpu_fallback")),
                "provider": provider,
            }
        else:
            sel = select_engine_for_inference(model, resolved, "auto", onnx_diag)
            comparison = {
                "recommended_engine": sel.get("selected_engine"),
                "recommendation_reason": sel.get("reason"),
                "recommendation_source": sel.get("source"),
                "display_label": "—",
                "selection_margin": ENGINE_SELECTION_MARGIN,
                "uses_cpu_fallback": bool(onnx_diag.get("runtime", {}).get("uses_cpu_fallback")),
                "provider": provider,
            }

        result = {
            "model": model,
            "model_id": model,
            "display_name": spec.display_name,
            "input_shape": [1, 3, *spec.input_size],
            "device_requested": device,
            "device_resolved": resolved,
            "iterations": iterations,
            "engine_requested": engine,
            "frame_shape": list(frame.shape),
            "weights": {
                "onnx_path": os.fspath(weights_path),
                "onnx_available": weights_path.exists(),
            },
            "onnx_diagnostics": onnx_diag,
            "results": [pytorch_result, onnx_result],
            "comparison": comparison,
            "speedup": comparison.get("speedup"),
            "pytorch": {
                "status": pytorch_result.get("status"),
                "latency_ms": (
                    (pytorch_result.get("latency_ms") or {}).get("avg")
                    if isinstance(pytorch_result.get("latency_ms"), dict)
                    else None
                ),
                "throughput_fps": pytorch_result.get("throughput_fps"),
                "device_used": resolved,
                "memory_mb": _memory_snapshot().get("process_rss_mb"),
                "error": pytorch_result.get("reason"),
            },
            "onnx": {
                "status": (
                    "ok"
                    if onnx_result.get("status") == "ok"
                    else onnx_result.get("state", "unavailable")
                ),
                "latency_ms": (
                    (onnx_result.get("latency_ms") or {}).get("avg")
                    if isinstance(onnx_result.get("latency_ms"), dict)
                    else None
                ),
                "throughput_fps": onnx_result.get("throughput_fps"),
                "providers_used": onnx_result.get("providers", []),
                "onnx_path": os.fspath(weights_path),
                "error_code": (
                    None
                    if onnx_result.get("status") == "ok"
                    else str(onnx_result.get("state", "unavailable")).upper()
                ),
                "message": onnx_result.get("reason"),
                "technical_detail": onnx_result.get("diagnostics", {}).get("export_error")
                or onnx_result.get("reason"),
            },
            "warnings": (
                []
                if onnx_result.get("status") == "ok"
                else [onnx_result.get("reason", "ONNX unavailable; PyTorch benchmark completed")]
            ),
            "memory_snapshot": _memory_snapshot(),
        }
        if (
            engine == "both"
            and pytorch_result.get("status") == "ok"
            and onnx_result.get("status") == "ok"
        ):
            record_benchmark_decision(result)
        pytorch_summary = cast(dict[str, Any], result.get("pytorch", {}))
        onnx_summary = cast(dict[str, Any], result.get("onnx", {}))
        providers_used = cast(list[str | None], onnx_summary.get("providers_used") or [None])
        warnings = cast(list[Any], result.get("warnings") or [])
        observability.record_benchmark(
            model,
            spec.display_name,
            device,
            resolved,
            iterations,
            cast(float | None, pytorch_summary.get("latency_ms")),
            cast(float | None, onnx_summary.get("latency_ms")),
            cast(float | None, result.get("speedup")),
            cast(str | None, onnx_summary.get("status")),
            providers_used[0],
            len(warnings),
            "ok",
        )
        log.info(
            "BENCHMARK_END model=%s device=%s onnx_state=%s",
            model,
            resolved,
            onnx_summary.get("status"),
        )
        return result
    finally:
        set_benchmark_busy(False)
