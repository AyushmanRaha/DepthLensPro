"""Local observability helpers for DepthLens Pro.

This module intentionally avoids ML/runtime imports so it can be imported from
FastAPI middleware, liveness-adjacent routes, cache code, and tests.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.config import settings

try:  # graceful optional dependency path
    from prometheus_client import (
        CollectorRegistry,
        Gauge,
        Histogram,
        generate_latest,
    )
    from prometheus_client import (
        Counter as PromCounter,
    )
    from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover - exercised by partial installs
    CollectorRegistry = None
    PromCounter = None
    Gauge = None
    Histogram = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

_LOCK = threading.RLock()
_STARTED = time.perf_counter()
_HOME = os.path.expanduser("~")
_ALLOWED = re.compile(r"[^A-Z0-9_]+")
_UNIX_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:/[^\s,;:'\")]+){2,}")
_WIN_PATH = re.compile(r"[A-Za-z]:\\\\[^\s,;:'\")]+")

inference_events: deque[dict[str, Any]]
request_events: deque[dict[str, Any]]
trace_events: deque[dict[str, Any]]
crash_events: deque[dict[str, Any]]
benchmark_history: deque[dict[str, Any]]
_http_total = 0
_http_errors = 0
_http_active = 0
_http_latencies: deque[float]
_inference_counts: Counter[str]
_inference_latencies: deque[float]
_cache_counts: Counter[str]
_cache_entries: int | None = None
_benchmark_total = 0
_error_counts: Counter[str]
_crash_counts: Counter[str]
_by_route: Counter[str]
_by_model: Counter[str]
_by_engine: Counter[str]

_registry: Any | None = None
_metrics: dict[str, Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enabled() -> bool:
    return bool(settings.DEPTHLENS_OBSERVABILITY_ENABLED)


def prometheus_enabled() -> bool:
    return enabled() and bool(settings.DEPTHLENS_PROMETHEUS_ENABLED) and generate_latest is not None


def _safe_label(value: Any, default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    text = re.sub(r"[^a-z0-9_./:-]+", "_", text)[:80]
    return text or default


def normalize_device_type(device: str | None) -> str:
    text = (device or "").lower()
    if "cpu" in text:
        return "cpu"
    if text.startswith("cuda") or "cuda" in text:
        return "cuda"
    if "mps" in text:
        return "mps"
    if "xpu" in text:
        return "xpu"
    if "npu" in text or "ane" in text:
        return "npu"
    return "unknown"


def sanitize_error_code(value: Any) -> str:
    token = _ALLOWED.sub("_", str(value or "UNKNOWN").upper()).strip("_")
    return token[:64] or "UNKNOWN"


def sanitize_message(value: Any) -> str:
    text = str(value or "")
    if _HOME and _HOME != "/":
        text = text.replace(_HOME, "[home]")
    text = _WIN_PATH.sub("[path]", text)
    text = _UNIX_PATH.sub("[path]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:240]


def _route(route: str) -> str:
    text = str(route or "unknown")
    if "://" in text:
        text = "/" + text.split("://", 1)[1].split("/", 1)[-1]
    text = text.split("?", 1)[0]
    return text[:120] or "unknown"


def _init_buffers() -> None:
    global inference_events, request_events, trace_events, crash_events, benchmark_history
    global _http_latencies, _inference_latencies
    inference_events = deque(maxlen=settings.DEPTHLENS_TELEMETRY_MAX_EVENTS)
    request_events = deque(maxlen=settings.DEPTHLENS_TELEMETRY_MAX_EVENTS)
    trace_events = deque(maxlen=settings.DEPTHLENS_TRACE_HISTORY_LIMIT)
    crash_events = deque(maxlen=settings.DEPTHLENS_CRASH_HISTORY_LIMIT)
    benchmark_history = deque(maxlen=settings.DEPTHLENS_BENCHMARK_HISTORY_LIMIT)
    _http_latencies = deque(maxlen=settings.DEPTHLENS_TELEMETRY_MAX_EVENTS)
    _inference_latencies = deque(maxlen=settings.DEPTHLENS_TELEMETRY_MAX_EVENTS)


def _init_prom() -> None:
    global _registry, _metrics
    _metrics = {}
    if CollectorRegistry is None:
        _registry = None
        return
    _registry = CollectorRegistry()
    buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
    _metrics = {
        "http_total": PromCounter(
            "depthlens_http_requests_total",
            "HTTP requests",
            ["method", "route", "status_class"],
            registry=_registry,
        ),
        "http_duration": Histogram(
            "depthlens_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "route", "status_class"],
            buckets=buckets,
            registry=_registry,
        ),
        "http_active": Gauge(
            "depthlens_http_requests_active", "Active HTTP requests", registry=_registry
        ),
        "inf_total": PromCounter(
            "depthlens_inference_requests_total",
            "Inference requests",
            ["model_id", "engine", "device_type", "outcome", "cached"],
            registry=_registry,
        ),
        "inf_latency": Histogram(
            "depthlens_inference_latency_seconds",
            "Inference latency",
            ["model_id", "engine", "device_type", "cached"],
            buckets=buckets,
            registry=_registry,
        ),
        "inf_pixels": PromCounter(
            "depthlens_inference_pixels_total",
            "Inference pixels",
            ["model_id", "engine", "device_type"],
            registry=_registry,
        ),
        "cache": PromCounter(
            "depthlens_cache_events_total", "Cache events", ["event", "backend"], registry=_registry
        ),
        "cache_entries": Gauge("depthlens_cache_entries", "Cache entries", registry=_registry),
        "bench": PromCounter(
            "depthlens_benchmark_runs_total",
            "Benchmark runs",
            ["model_id", "device_type", "outcome"],
            registry=_registry,
        ),
        "bench_latency": Histogram(
            "depthlens_benchmark_latency_seconds",
            "Benchmark latency",
            ["model_id", "engine", "device_type"],
            buckets=buckets,
            registry=_registry,
        ),
        "errors": PromCounter(
            "depthlens_errors_total", "Errors", ["component", "error_code"], registry=_registry
        ),
        "crashes": PromCounter(
            "depthlens_crashes_total", "Crashes", ["component", "error_code"], registry=_registry
        ),
        "traces": PromCounter(
            "depthlens_trace_spans_total",
            "Trace spans",
            ["component", "span", "outcome"],
            registry=_registry,
        ),
        "trace_duration": Histogram(
            "depthlens_trace_span_duration_seconds",
            "Trace span duration",
            ["component", "span", "outcome"],
            buckets=buckets,
            registry=_registry,
        ),
    }


def reset_for_tests() -> None:
    global _http_total, _http_errors, _http_active, _cache_entries, _benchmark_total
    global _inference_counts, _cache_counts, _error_counts, _crash_counts
    global _by_route, _by_model, _by_engine
    with _LOCK:
        _init_buffers()
        _http_total = _http_errors = _http_active = _benchmark_total = 0
        _cache_entries = None
        _inference_counts = Counter()
        _cache_counts = Counter()
        _error_counts = Counter()
        _crash_counts = Counter()
        _by_route = Counter()
        _by_model = Counter()
        _by_engine = Counter()
        _init_prom()


reset_for_tests()


def increment_active_http() -> None:
    global _http_active
    if not enabled():
        return
    with _LOCK:
        _http_active += 1
        if prometheus_enabled():
            _metrics["http_active"].inc()


def decrement_active_http() -> None:
    global _http_active
    if not enabled():
        return
    with _LOCK:
        _http_active = max(0, _http_active - 1)
        if prometheus_enabled():
            _metrics["http_active"].dec()


def observe_http_request(method: str, route: str, status_code: int, duration_s: float) -> None:
    global _http_total, _http_errors
    if not enabled():
        return
    status_class = f"{int(status_code) // 100}xx"
    route_label = _route(route)
    duration_ms = round(max(duration_s, 0.0) * 1000, 2)
    with _LOCK:
        _http_total += 1
        if status_code >= 500:
            _http_errors += 1
        _http_latencies.append(duration_ms)
        _by_route[route_label] += 1
        request_events.append(
            {
                "timestamp": _now(),
                "method": method.upper(),
                "route": route_label,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
        )
        if prometheus_enabled():
            _metrics["http_total"].labels(method.upper(), route_label, status_class).inc()
            _metrics["http_duration"].labels(method.upper(), route_label, status_class).observe(
                max(duration_s, 0.0)
            )


@contextmanager
def trace_span(component: str, span: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
    if not enabled() or random.random() > settings.DEPTHLENS_TRACE_SAMPLE_RATE:
        yield
        return
    started = time.perf_counter()
    outcome = "ok"
    trace_id = str(uuid.uuid4())
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        safe_meta = _sanitize_metadata(metadata or {})
        comp = _safe_label(component)
        sp = _safe_label(span)
        with _LOCK:
            trace_events.append(
                {
                    "trace_id": trace_id,
                    "timestamp": _now(),
                    "component": comp,
                    "span": sp,
                    "duration_ms": duration_ms,
                    "outcome": outcome,
                    "metadata": safe_meta,
                }
            )
            if prometheus_enabled():
                _metrics["traces"].labels(comp, sp, outcome).inc()
                _metrics["trace_duration"].labels(comp, sp, outcome).observe(duration_ms / 1000)


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "model_id",
        "engine",
        "device_type",
        "cached",
        "metrics_mode",
        "outputs_count",
        "warnings_count",
    ):
        if key in meta:
            out[key] = (
                meta[key] if isinstance(meta[key], (bool, int, float)) else _safe_label(meta[key])
            )
    return out


def record_inference(
    model_id: str,
    engine: str,
    device: str,
    latency_ms: float | int | None,
    pixels: int | None = None,
    cached: bool = False,
    outcome: str = "ok",
    metrics_mode: str | None = None,
    outputs_count: int | None = None,
    warnings_count: int = 0,
    error_code: str | None = None,
) -> None:
    if not enabled():
        return
    dev = normalize_device_type(device)
    model_label = _safe_label(model_id)
    engine_label = _safe_label(engine)
    outcome_label = "ok" if outcome == "ok" else "error"
    lat = float(latency_ms) if isinstance(latency_ms, (int, float)) else None
    with _LOCK:
        _inference_counts["total"] += 1
        _inference_counts["success" if outcome_label == "ok" else "failed"] += 1
        if cached:
            _inference_counts["cache_served"] += 1
        if lat is not None:
            _inference_latencies.append(lat)
        _by_model[model_label] += 1
        _by_engine[engine_label] += 1
        inference_events.append(
            {
                "timestamp": _now(),
                "model_id": model_label,
                "engine": engine_label,
                "device_type": dev,
                "latency_ms": lat,
                "pixels": pixels,
                "cached": bool(cached),
                "outcome": outcome_label,
                "metrics_mode": metrics_mode,
                "outputs_count": outputs_count,
                "warnings_count": warnings_count,
                "error_code": sanitize_error_code(error_code) if error_code else None,
            }
        )
        if prometheus_enabled():
            _metrics["inf_total"].labels(
                model_label, engine_label, dev, outcome_label, str(bool(cached)).lower()
            ).inc()
            if lat is not None:
                _metrics["inf_latency"].labels(
                    model_label, engine_label, dev, str(bool(cached)).lower()
                ).observe(lat / 1000)
            if pixels:
                _metrics["inf_pixels"].labels(model_label, engine_label, dev).inc(float(pixels))


def record_cache_event(event: str, backend: str = "unknown", entries: int | None = None) -> None:
    global _cache_entries
    if not enabled():
        return
    ev = _safe_label(event)
    be = _safe_label(backend)
    with _LOCK:
        _cache_counts[ev] += 1
        if entries is not None:
            _cache_entries = max(0, int(entries))
        if prometheus_enabled():
            _metrics["cache"].labels(ev, be).inc()
            if entries is not None:
                _metrics["cache_entries"].set(max(0, int(entries)))


def record_benchmark(
    model_id: str,
    display_name: str | None,
    device_requested: str,
    device_resolved: str,
    iterations: int,
    pytorch_latency_ms: float | None,
    onnx_latency_ms: float | None,
    speedup: float | None,
    onnx_status: str | None,
    provider: str | None,
    warnings_count: int,
    outcome: str,
    total_latency_ms: float | None = None,
) -> None:
    global _benchmark_total
    if not enabled():
        return
    dev = normalize_device_type(device_resolved or device_requested)
    model_label = _safe_label(model_id)
    outcome_label = "ok" if outcome == "ok" else "error"
    event = {
        "timestamp": _now(),
        "model_id": model_label,
        "display_name": sanitize_message(display_name or model_label),
        "device_requested": _safe_label(device_requested),
        "device_resolved": _safe_label(device_resolved),
        "device_type": dev,
        "iterations": iterations,
        "pytorch_latency_ms": pytorch_latency_ms,
        "onnx_latency_ms": onnx_latency_ms,
        "speedup": speedup,
        "onnx_status": _safe_label(onnx_status),
        "provider": _safe_label(provider),
        "warnings_count": warnings_count,
        "outcome": outcome_label,
        "total_latency_ms": total_latency_ms,
    }
    with _LOCK:
        _benchmark_total += 1
        benchmark_history.append(event)
        if prometheus_enabled():
            _metrics["bench"].labels(model_label, dev, outcome_label).inc()
            if pytorch_latency_ms is not None:
                _metrics["bench_latency"].labels(model_label, "pytorch", dev).observe(
                    float(pytorch_latency_ms) / 1000
                )
            if onnx_latency_ms is not None:
                _metrics["bench_latency"].labels(model_label, "onnxruntime", dev).observe(
                    float(onnx_latency_ms) / 1000
                )


def record_error(component: str, error_code: str, exc: Exception | None = None) -> None:
    if not enabled():
        return
    comp = _safe_label(component)
    code = sanitize_error_code(error_code)
    with _LOCK:
        _error_counts[f"{comp}:{code}"] += 1
        if prometheus_enabled():
            _metrics["errors"].labels(comp, code).inc()


def record_crash(
    component: str, error_code: str, exc: Exception | None = None, route: str | None = None
) -> None:
    if not enabled():
        return
    record_error(component, error_code, exc)
    comp = _safe_label(component)
    code = sanitize_error_code(error_code)
    with _LOCK:
        _crash_counts[f"{comp}:{code}"] += 1
        crash_events.append(
            {
                "timestamp": _now(),
                "component": comp,
                "error_code": code,
                "exception_type": type(exc).__name__ if exc else None,
                "message": sanitize_message(exc),
                "route": _route(route or ""),
                "trace_id": str(uuid.uuid4()),
            }
        )
        if prometheus_enabled():
            _metrics["crashes"].labels(comp, code).inc()


def _pct(vals: deque[float], pct: float) -> float | None:
    if not vals:
        return None
    data = sorted(vals)
    idx = min(len(data) - 1, max(0, int(round((pct / 100) * (len(data) - 1)))))
    return round(data[idx], 2)


def _avg(vals: deque[float]) -> float | None:
    return round(sum(vals) / len(vals), 2) if vals else None


def snapshot() -> dict[str, Any]:
    with _LOCK:
        hits = _cache_counts.get("hit", 0)
        misses = _cache_counts.get("miss", 0)
        denom = hits + misses
        return {
            "status": "ok" if enabled() else "disabled",
            "enabled": enabled(),
            "prometheus_enabled": prometheus_enabled(),
            "generated_at": _now(),
            "process": {
                "pid": os.getpid(),
                "uptime_seconds": round(time.perf_counter() - _STARTED, 2),
            },
            "http": {
                "total_requests": _http_total,
                "error_requests": _http_errors,
                "active_requests": _http_active,
                "avg_latency_ms": _avg(_http_latencies),
                "p50_latency_ms": _pct(_http_latencies, 50),
                "p95_latency_ms": _pct(_http_latencies, 95),
                "by_route": [{"route": k, "count": v} for k, v in _by_route.most_common()],
                "recent": list(request_events)[-25:],
            },
            "inference": {
                "total": _inference_counts.get("total", 0),
                "success": _inference_counts.get("success", 0),
                "failed": _inference_counts.get("failed", 0),
                "cache_served": _inference_counts.get("cache_served", 0),
                "avg_latency_ms": _avg(_inference_latencies),
                "p50_latency_ms": _pct(_inference_latencies, 50),
                "p95_latency_ms": _pct(_inference_latencies, 95),
                "by_model": dict(_by_model),
                "by_engine": dict(_by_engine),
                "recent": list(inference_events)[-50:],
            },
            "cache": {
                "events": dict(_cache_counts),
                "entries": _cache_entries,
                "hit_ratio_percent": round(hits / denom * 100, 2) if denom else None,
            },
            "traces": {
                "recent": list(trace_events)[-50:],
                "slowest": sorted(
                    trace_events, key=lambda e: e.get("duration_ms") or 0, reverse=True
                )[:10],
            },
            "crashes": {"total": sum(_crash_counts.values()), "recent": list(crash_events)[-25:]},
            "benchmarks": {"total": _benchmark_total, "history": list(benchmark_history)[-50:]},
        }


def prometheus_text() -> tuple[str, str]:
    if not enabled():
        return "# DepthLens observability disabled\n", CONTENT_TYPE_LATEST
    if not prometheus_enabled() or _registry is None:
        return "# DepthLens Prometheus disabled or unavailable\n", CONTENT_TYPE_LATEST
    return generate_latest(_registry).decode("utf-8"), CONTENT_TYPE_LATEST
