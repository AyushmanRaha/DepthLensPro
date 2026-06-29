"""Adaptive PyTorch/ONNX Runtime engine selection."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any

from backend.model_registry import normalize_model_id, resolve_onnx_path
from backend.services.onnx_diagnostics import create_onnx_session, onnx_runtime_info

ENGINE_SELECTION_MARGIN = 0.90
ENGINE_CHOICES = {"auto", "pytorch", "onnxruntime"}
_SERVICE_VERSION = "engine-selector-v1"
_DECISIONS: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()


def normalize_engine_mode(value: str | None, *, allow_both: bool = False) -> str:
    mode = (value or "auto").strip().lower().replace("-", "_")
    aliases = {"onnx": "onnxruntime", "ort": "onnxruntime"}
    if allow_both:
        aliases.update({"compare": "both"})
    mode = aliases.get(mode, mode)
    allowed = set(ENGINE_CHOICES) | ({"both"} if allow_both else set())
    if mode not in allowed:
        raise ValueError(f"engine must be one of: {', '.join(sorted(allowed))}")
    return mode


@dataclass(frozen=True)
class EngineDecision:
    selected_engine: str
    source: str
    reason: str
    requested_engine: str = "auto"
    provider_signature: str | None = None
    onnx_file_hash: str | None = None
    recommended_engine: str | None = None
    pytorch_latency_ms: float | None = None
    onnx_latency_ms: float | None = None
    selection_margin: float = ENGINE_SELECTION_MARGIN

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["engine"] = self.selected_engine
        return data


def onnx_file_fingerprint(model_id: str) -> str | None:
    detail = resolve_onnx_path(normalize_model_id(model_id))
    path = detail.get("onnx_path")
    if not path or not os.path.isfile(str(path)):
        return None
    stat = os.stat(str(path))
    raw = f"{os.path.abspath(str(path))}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def provider_signature(device: str, model_id: str) -> str:
    info = onnx_runtime_info(device)
    selected = info.get("selected_providers") or []
    return "+".join(str(p) for p in selected) or str(info.get("provider_state") or "unknown")


def decision_key(
    model_id: str,
    device: str,
    provider_sig: str,
    onnx_file_hash: str | None,
    service_version: str = _SERVICE_VERSION,
) -> str:
    return "|".join(
        [
            normalize_model_id(model_id),
            str(device),
            provider_sig,
            onnx_file_hash or "none",
            service_version,
        ]
    )


def _onnx_healthy(onnx_detail: dict[str, Any] | None, model_id: str | None = None) -> bool:
    if onnx_detail is not None:
        return onnx_detail.get("state") == "available" or bool(onnx_detail.get("session_available"))
    if not model_id:
        return False
    detail = resolve_onnx_path(normalize_model_id(model_id))
    return bool(detail.get("exists") and int(detail.get("size_bytes") or 0) > 0)


def _onnx_state(model_id: str, device: str, onnx_detail: dict[str, Any] | None) -> dict[str, Any]:
    detail = onnx_detail or resolve_onnx_path(model_id)
    asset_available = bool(detail.get("exists") and int(detail.get("size_bytes") or 0) > 0)
    runtime = onnx_runtime_info(device)
    runtime_importable = bool(runtime.get("importable"))
    providers = list(runtime.get("selected_providers") or runtime.get("available_providers") or [])
    session_result: dict[str, Any] = {}
    if asset_available and runtime_importable and providers:
        session_result = create_onnx_session(model_id, device, model_path=detail.get("onnx_path"))
    return {
        "asset_available": asset_available,
        "runtime_importable": runtime_importable,
        "provider_available": bool(providers),
        "session_available": bool(session_result.get("ok")),
        "providers_attempted": session_result.get("providers_attempted") or [providers],
        "providers_used": session_result.get("providers_used") or [],
        "provider_fallback_used": bool(session_result.get("provider_fallback_used")),
        "onnx_failure_reason": session_result.get("technical_detail")
        or session_result.get("message")
        or detail.get("error"),
        "provider_failure_chain": session_result.get("provider_failure_chain") or [],
    }


def get_cached_decision(model_id: str, device: str) -> dict[str, Any] | None:
    model_id = normalize_model_id(model_id)
    sig = provider_signature(device, model_id)
    fp = onnx_file_fingerprint(model_id)
    key = decision_key(model_id, device, sig, fp)
    with _LOCK:
        decision = _DECISIONS.get(key)
        return dict(decision) if decision else None


def clear_engine_decisions() -> None:
    with _LOCK:
        _DECISIONS.clear()


def select_engine_for_inference(
    model_id: str, device: str, requested: str | None, onnx_detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    model_id = normalize_model_id(model_id)
    mode = normalize_engine_mode(requested)
    sig = provider_signature(device, model_id)
    fp = onnx_file_fingerprint(model_id)
    state = _onnx_state(model_id, device, onnx_detail)

    def decision(
        selected: str, source: str, reason: str, fallback_target: str | None = None
    ) -> dict[str, Any]:
        data = EngineDecision(selected, source, reason, mode, sig, fp).to_dict()
        data.update(state)
        data.update(
            {
                "fallback_target": fallback_target,
                "onnx_file_hash": fp,
                "provider_signature": sig,
            }
        )
        return data

    if mode == "pytorch":
        return decision("pytorch", "forced", "PyTorch forced by request")
    if mode == "onnxruntime":
        if state["session_available"]:
            reason = "ONNX Runtime forced by request"
            if state["provider_fallback_used"]:
                reason += "; preferred provider failed, using ONNX CPU provider fallback"
            return decision("onnxruntime", "forced", reason)
        reason = state["onnx_failure_reason"] or "ONNX Runtime session unavailable"
        return decision("pytorch", "forced_fallback", reason, "pytorch")

    key = decision_key(model_id, device, sig, fp)
    with _LOCK:
        cached = _DECISIONS.get(key)
    if cached and state["session_available"]:
        return {**cached, **state, "requested_engine": mode, "source": "cached_benchmark"}
    if state["session_available"] and model_id == "midas_small":
        return decision(
            "onnxruntime", "conservative_default", "Verified ONNX session selected for MiDaS Small"
        )
    if state["asset_available"] and not state["session_available"]:
        return decision(
            "pytorch",
            "conservative_default",
            "ONNX asset present but session unverified; PyTorch selected",
        )
    return decision("pytorch", "conservative_default", "ONNX unavailable; PyTorch selected")


def record_benchmark_decision(benchmark_result: dict[str, Any]) -> dict[str, Any]:
    model_id = normalize_model_id(
        str(benchmark_result.get("model_id") or benchmark_result.get("model") or "MiDaS_small")
    )
    device = str(
        benchmark_result.get("device_resolved")
        or benchmark_result.get("device_requested")
        or "auto"
    )
    pt = (benchmark_result.get("pytorch") or {}).get("latency_ms")
    ox = (benchmark_result.get("onnx") or {}).get("latency_ms")
    try:
        if pt is None or ox is None:
            raise ValueError("missing latency")
        pt_f, ox_f = float(pt), float(ox)
    except (TypeError, ValueError):
        return {}
    sig = provider_signature(device, model_id)
    fp = onnx_file_fingerprint(model_id)
    if ox_f <= pt_f * ENGINE_SELECTION_MARGIN:
        selected = "onnxruntime"
        factor = pt_f / ox_f if ox_f else 0
        reason = f"ONNX Runtime was faster by {factor:.1f}× in the latest benchmark"
    else:
        selected = "pytorch"
        reason = "PyTorch was faster in the latest benchmark"
    decision = EngineDecision(
        selected, "cached_benchmark", reason, "auto", sig, fp, selected, pt_f, ox_f
    ).to_dict()
    with _LOCK:
        _DECISIONS[decision_key(model_id, device, sig, fp)] = decision
    return dict(decision)
