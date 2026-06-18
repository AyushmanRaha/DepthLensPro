"""ONNX Runtime, static weight, and guarded session diagnostics."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any

from backend.model_registry import MODEL_REGISTRY, get_model_spec, resolve_onnx_path
from backend.utils.hardware import (
    _default_device_key,
    _onnx_provider_candidates,
    _onnx_providers_for_device,
)


def export_command(model: str) -> str:
    return f"python backend/scripts/export_onnx.py --model {model} --force"


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


def _explicit_path_payload(model: str, model_path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        spec = get_model_spec(model)
    except Exception:
        display_name = None
        model_id = model
    else:
        display_name = spec.display_name
        model_id = spec.model_id
    path = Path(model_path).expanduser().resolve()
    exists = path.is_file()
    return {
        "model_id": model_id,
        "display_name": display_name,
        "onnx_path": os.fspath(path),
        "expected_path": os.fspath(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "source": "explicit",
        "error": None if exists else "missing_file",
        "candidates": [os.fspath(path)],
    }


def create_onnx_session(
    model: str,
    device: str = "auto",
    *,
    model_path: str | os.PathLike[str] | None = None,
    session_options: Any | None = None,
) -> dict[str, Any]:
    """Safely create an ONNX Runtime session only after validating path/providers."""

    resolved = (
        _explicit_path_payload(model, model_path)
        if model_path is not None
        else resolve_onnx_path(model)
    )
    try:
        spec = get_model_spec(model)
    except Exception:
        return {
            "ok": False,
            "error_code": "UNKNOWN_MODEL",
            "message": f"Unknown model id: {model}",
            "valid_models": list(MODEL_REGISTRY),
            "fallback_allowed": False,
            "path": resolved,
        }

    path = resolved.get("onnx_path")
    if not path or not isinstance(path, str):
        return {
            "ok": False,
            "error_code": "ONNX_MODEL_MISSING",
            "message": f"ONNX file missing for {spec.display_name}; PyTorch fallback is available.",
            "expected_path": resolved.get("expected_path"),
            "fallback_allowed": True,
            "path": resolved,
        }
    if not resolved.get("exists") or not os.path.isfile(path):
        return {
            "ok": False,
            "error_code": "ONNX_MODEL_MISSING",
            "message": f"ONNX file missing for {spec.display_name}; PyTorch fallback is available.",
            "expected_path": resolved.get("expected_path") or path,
            "fallback_allowed": True,
            "path": resolved,
        }
    if int(resolved.get("size_bytes") or 0) <= 0:
        return {
            "ok": False,
            "error_code": "ONNX_MODEL_EMPTY",
            "message": (
                f"ONNX file is empty for {spec.display_name}; " "PyTorch fallback is available."
            ),
            "expected_path": path,
            "fallback_allowed": True,
            "path": resolved,
        }

    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:
        return {
            "ok": False,
            "error_code": "ONNXRUNTIME_MISSING",
            "message": (
                "onnxruntime is not installed or cannot be imported; "
                "PyTorch fallback is available."
            ),
            "technical_detail": f"{type(exc).__name__}: {exc}",
            "fallback_allowed": True,
            "path": resolved,
        }

    available = list(ort.get_available_providers())
    providers = _onnx_providers_for_device(device, available)
    if not providers:
        return {
            "ok": False,
            "error_code": "ONNX_PROVIDER_UNAVAILABLE",
            "message": (
                "No compatible ONNX Runtime execution provider is available; "
                "PyTorch fallback is available."
            ),
            "available_providers": available,
            "requested_device": device,
            "fallback_allowed": True,
            "path": resolved,
        }

    try:
        session = ort.InferenceSession(path, sess_options=session_options, providers=providers)
        return {
            "ok": True,
            "session": session,
            "providers_used": providers,
            "available_providers": available,
            "path": resolved,
            "model_id": spec.model_id,
            "display_name": spec.display_name,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_code": "ONNX_SESSION_INIT_FAILED",
            "message": (
                f"ONNX Runtime could not load {spec.display_name} from {path}; "
                "PyTorch fallback is available."
            ),
            "technical_detail": f"{type(exc).__name__}: {exc}",
            "providers_used": providers,
            "available_providers": available,
            "fallback_allowed": True,
            "path": resolved,
        }


def _checker_status(path: str | None) -> tuple[str | None, str | None]:
    if not path:
        return None, None
    p = Path(path)
    if not p.exists():
        return "missing", None
    if p.stat().st_size <= 0:
        return "empty", "ONNX file is empty"
    try:
        onnx = importlib.import_module("onnx")
    except Exception:
        return None, None
    try:
        model = onnx.load(os.fspath(p), load_external_data=True)
        onnx.checker.check_model(model)
        return "checker_ok", None
    except Exception as exc:
        return "invalid_checker", f"{type(exc).__name__}: {exc}"


def _status_from_session_result(
    session_result: dict[str, Any], runtime: dict[str, Any], checker_state: str | None = None
) -> str:
    if session_result.get("ok"):
        return "available"
    error_code = session_result.get("error_code")
    if error_code in {"ONNX_MODEL_MISSING", "ONNX_MODEL_PATH_EMPTY"}:
        return "missing"
    if error_code == "ONNX_MODEL_EMPTY":
        return "empty"
    if checker_state == "invalid_checker":
        return "invalid_checker"
    if error_code == "ONNX_SESSION_INIT_FAILED":
        return "invalid_session"
    if error_code == "ONNXRUNTIME_MISSING" or not runtime.get("importable"):
        return "runtime_unavailable"
    if error_code == "ONNX_PROVIDER_UNAVAILABLE":
        return "provider_unavailable"
    return "invalid_session"


def onnx_model_status(model: str, device: str = "auto") -> dict[str, Any]:
    resolved = resolve_onnx_path(model)
    runtime = onnx_runtime_info(device)
    if resolved.get("error") == "unknown_model":
        state = "unknown_model"
        session_result: dict[str, Any] = {"ok": False, "error_code": "UNKNOWN_MODEL"}
    else:
        if resolved.get("onnx_path"):
            session_result = create_onnx_session(
                model, device, model_path=resolved.get("onnx_path")
            )
        else:
            session_result = create_onnx_session(model, device)
        checker_state, checker_error = _checker_status(resolved.get("onnx_path"))
        state = _status_from_session_result(session_result, runtime, checker_state)
    try:
        optional_onnx = bool(
            get_model_spec(resolved.get("model_id") or model).model_id
            in {"dpt_hybrid", "dpt_large"}
        )
    except Exception:
        optional_onnx = False
    if optional_onnx and state in {"missing", "export_failed"}:
        state = "optional_unavailable"
    return {
        "model": resolved.get("model_id") or model,
        "display_name": resolved.get("display_name"),
        "expected_path": resolved.get("expected_path") or resolved.get("onnx_path"),
        "selected_path": resolved.get("onnx_path"),
        "exists": bool(resolved.get("exists")),
        "size_bytes": resolved.get("size_bytes") or 0,
        "state": state,
        "legacy_state": (
            "invalid/corrupt"
            if state in {"invalid_checker", "invalid_session", "invalid_dummy_inference"}
            else state
        ),
        "checker_error": locals().get("checker_error"),
        "input_shape": (
            [1, 3, *get_model_spec(resolved.get("model_id") or model).input_size]
            if (resolved.get("model_id") or model) in MODEL_REGISTRY
            else None
        ),
        "fallback_behavior": "PyTorch fallback remains available",
        "path": resolved,
        "providers_used": session_result.get("providers_used", []),
        "available_providers": session_result.get(
            "available_providers", runtime.get("available_providers", [])
        ),
        "error_code": session_result.get("error_code"),
        "message": session_result.get("message"),
        "technical_detail": session_result.get("technical_detail"),
        "recommended_export_command": (
            None if state == "available" else export_command(str(resolved.get("model_id") or model))
        ),
        "runtime": runtime,
    }


def onnx_status_payload(device: str = "auto") -> dict[str, Any]:
    runtime = onnx_runtime_info(device)
    models = {model: onnx_model_status(model, device) for model in MODEL_REGISTRY}
    onnx_ready = any(m.get("state") == "available" for m in models.values())
    return {
        "supported_model_ids": list(MODEL_REGISTRY),
        "requested_device": device,
        "runtime": runtime,
        "models": models,
        "overall_status": "onnx_ready" if onnx_ready else "onnx_unavailable",
    }


def readiness_payload(device: str = "auto") -> dict[str, Any]:
    """Detailed backend/model/device readiness for health endpoints."""

    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(
            mps_backend is not None and mps_backend.is_built() and mps_backend.is_available()
        )
    except Exception:
        torch_version = None
        cuda = False
        mps = False

    selected = _default_device_key()
    onnx = onnx_status_payload(device if device != "auto" else selected)
    models = {}
    onnx_any = False
    for model_id, spec in MODEL_REGISTRY.items():
        status = onnx["models"][model_id]
        ready = status.get("state") == "available"
        onnx_any = onnx_any or ready
        models[model_id] = {
            "display_name": spec.display_name,
            "pytorch_ready": spec.pytorch_supported,
            "onnx_ready": ready,
            "onnx_path": status.get("expected_path"),
            "onnx_error": None if ready else status.get("state"),
        }
    warnings = [] if onnx_any else ["ONNX models are unavailable. PyTorch fallback will be used."]
    return {
        "backend_live": True,
        "platform": {
            "os": sys.platform,
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch_version,
            "onnxruntime": onnx["runtime"].get("version"),
        },
        "devices": {
            "pytorch": {"cpu": True, "cuda": cuda, "mps": mps, "selected": selected},
            "onnxruntime": {
                "available_providers": onnx["runtime"].get("available_providers", []),
                "selected_providers": onnx["runtime"].get("selected_providers", []),
            },
        },
        "models": models,
        "overall_status": "onnx_ready" if onnx_any else "pytorch_ready_onnx_unavailable",
        "warnings": warnings,
    }
