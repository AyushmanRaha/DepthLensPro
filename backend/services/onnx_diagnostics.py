"""ONNX Runtime and static weight diagnostics."""

from __future__ import annotations

import importlib
import os
from typing import Any

from backend.depth_models import onnx_model_path
from backend.model_metadata import SUPPORTED_MODELS
from backend.utils.hardware import _onnx_provider_candidates, _onnx_providers_for_device


def export_command(model: str) -> str:
    return f"python backend/scripts/export_onnx.py --model {model}"


def onnx_runtime_info(device: str = "auto") -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False,
        "available_providers": [],
        "requested_device": device,
        "provider_candidates": _onnx_provider_candidates(device),
        "selected_providers": [],
        "selected_provider": None,
        "provider_state": "onnxruntime_missing",
    }
    try:
        ort = importlib.import_module("onnxruntime")
        providers = list(ort.get_available_providers())
        selected = _onnx_providers_for_device(device, providers)
        candidates = _onnx_provider_candidates(device)
        used_fallback = bool(selected and candidates and selected[0] != candidates[0])
        info.update(
            {
                "importable": True,
                "version": getattr(ort, "__version__", None),
                "available_providers": providers,
                "selected_providers": selected,
                "selected_provider": selected[0] if selected else None,
                "provider_state": "available" if selected else "provider_unavailable",
                "uses_cpu_fallback": bool(selected == ["CPUExecutionProvider"] or used_fallback),
                "missing_preferred_providers": [p for p in candidates if p not in providers],
            }
        )
    except Exception as exc:
        info.update({"error": f"{type(exc).__name__}: {exc}"})
    return info


def onnx_model_status(model: str, device: str = "auto") -> dict[str, Any]:
    path = onnx_model_path(model)
    runtime = onnx_runtime_info(device)
    exists = path.exists()
    state = "available"
    if not runtime.get("importable"):
        state = "onnxruntime_missing"
    elif not exists:
        state = "missing_weights"
    elif runtime.get("provider_state") == "provider_unavailable":
        state = "provider_unavailable"
    return {
        "model": model,
        "expected_path": os.fspath(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "state": state,
        "recommended_export_command": None if exists else export_command(model),
        "runtime": runtime,
    }


def onnx_status_payload(device: str = "auto") -> dict[str, Any]:
    return {
        "supported_model_ids": list(SUPPORTED_MODELS),
        "requested_device": device,
        "runtime": onnx_runtime_info(device),
        "models": {model: onnx_model_status(model, device) for model in SUPPORTED_MODELS},
    }
