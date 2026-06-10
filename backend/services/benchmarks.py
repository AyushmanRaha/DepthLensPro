"""Benchmark helpers for comparing PyTorch and ONNX Runtime inference paths."""

from __future__ import annotations

import os
import platform
import resource
import threading
import time
from statistics import mean
from typing import Any, Callable

import numpy as np

from backend.depth_models import onnx_model_path
from backend.model_registry import get_model_spec, normalize_model_id
from backend.services.inference import _infer_onnx, _infer_torch
from backend.services.onnx_diagnostics import onnx_model_status
from backend.utils.hardware import _resolve

DEFAULT_BENCHMARK_ITERATIONS = 3
_AUTO_EXPORT_LOCKS: dict[str, threading.Lock] = {}
_AUTO_EXPORT_LOCKS_GUARD = threading.Lock()


def _env_flag(name: str, default: bool = True) -> bool:
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

    exportable_states = {"missing_weights", "missing_model_file", "missing", "invalid/corrupt"}
    if onnx_diag.get("state") not in exportable_states:
        return onnx_diag
    if not _env_flag("DEPTHLENS_AUTO_EXPORT_ONNX", True):
        return onnx_diag

    requested_device = str(onnx_diag.get("runtime", {}).get("requested_device", "auto"))
    with _auto_export_lock(model):
        refreshed = onnx_model_status(model, requested_device)
        if refreshed.get("state") not in exportable_states:
            return refreshed

        if refreshed.get("state") == "invalid/corrupt":
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
    model: str = "MiDaS_small", device: str = "auto", iterations: int = DEFAULT_BENCHMARK_ITERATIONS
) -> dict[str, Any]:
    """Return PyTorch-vs-ONNX latency, throughput, and memory benchmark matrices."""

    model = normalize_model_id(model)
    spec = get_model_spec(model)
    iterations = max(1, min(iterations, 20))
    resolved = str(_resolve(device))
    frame = _synthetic_frame()

    pytorch_result: dict[str, Any]
    onnx_result: dict[str, Any]

    try:
        pytorch_result = _measure(
            "pytorch", lambda: _infer_torch(frame, model, resolved), iterations
        )
    except Exception as exc:
        pytorch_result = _unavailable("pytorch", str(exc))

    weights_path = onnx_model_path(model)
    onnx_diag = _ensure_onnx_weights(model, onnx_model_status(model, resolved))
    if onnx_diag["state"] == "available":
        try:
            onnx_result = _measure(
                "onnxruntime", lambda: _infer_onnx(frame, model, resolved), iterations
            )
            onnx_result["provider"] = onnx_diag["runtime"].get("selected_provider")
            onnx_result["providers"] = onnx_diag["runtime"].get("selected_providers", [])
            onnx_result["uses_cpu_fallback"] = onnx_diag["runtime"].get("uses_cpu_fallback", False)
        except Exception as exc:
            onnx_result = _unavailable(
                "onnxruntime", str(exc), "runtime_error", {"diagnostics": onnx_diag}
            )
    elif onnx_diag["state"] in {"missing_weights", "missing_model_file", "missing"}:
        command = onnx_diag["recommended_export_command"]
        onnx_result = _unavailable(
            "onnxruntime",
            f"ONNX weights missing. Expected {weights_path}. Run {command}.",
            "missing_model_file",
            {"diagnostics": onnx_diag},
        )
    elif onnx_diag["state"] in {"onnxruntime_missing", "runtime_unavailable"}:
        onnx_result = _unavailable(
            "onnxruntime",
            "onnxruntime is not installed or cannot be imported",
            "onnxruntime_missing",
            {"diagnostics": onnx_diag},
        )
    elif onnx_diag["state"] == "provider_unavailable":
        onnx_result = _unavailable(
            "onnxruntime",
            "No compatible ONNX Runtime execution provider is available",
            "provider_unavailable",
            {"diagnostics": onnx_diag},
        )
    elif onnx_diag["state"] == "invalid/corrupt":
        onnx_result = _unavailable(
            "onnxruntime",
            "ONNX weights are invalid or corrupt",
            "invalid/corrupt",
            {"diagnostics": onnx_diag},
        )
    elif onnx_diag["state"] == "export_failed":
        onnx_result = _unavailable(
            "onnxruntime",
            f"Automatic ONNX export failed: {onnx_diag.get('export_error', 'unknown error')}",
            "export_failed",
            {"diagnostics": onnx_diag},
        )
    else:
        onnx_result = _unavailable(
            "onnxruntime", "ONNX Runtime is unavailable", "unavailable", {"diagnostics": onnx_diag}
        )

    comparison: dict[str, Any] = {}
    pt_avg = (pytorch_result.get("latency_ms") or {}).get("avg")
    onnx_avg = (onnx_result.get("latency_ms") or {}).get("avg")
    if isinstance(pt_avg, (int, float)) and isinstance(onnx_avg, (int, float)) and onnx_avg > 0:
        comparison = {
            "latency_delta_ms": round(pt_avg - onnx_avg, 2),
            "speedup": round(pt_avg / onnx_avg, 2),
            "faster_engine": "onnxruntime" if onnx_avg < pt_avg else "pytorch",
        }

    return {
        "model": model,
        "model_id": model,
        "display_name": spec.display_name,
        "input_shape": [1, 3, *spec.input_size],
        "device_requested": device,
        "device_resolved": resolved,
        "iterations": iterations,
        "frame_shape": list(frame.shape),
        "weights": {"onnx_path": os.fspath(weights_path), "onnx_available": weights_path.exists()},
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
