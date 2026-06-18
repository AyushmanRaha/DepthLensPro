"""Runtime readiness diagnostics for packaging and renderer startup checks."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
import time
from typing import Any

from backend.config import settings
from backend.model_metadata import COLORMAP_NAMES, SUPPORTED_MODELS
from backend.services.model_assets import inspect_model_assets
from backend.services.onnx_diagnostics import onnx_status_payload

REQUIRED_RUNTIME_MODULES = ("fastapi", "uvicorn", "numpy", "torch", "cv2", "PIL")
OPTIONAL_RUNTIME_MODULES = ("onnxruntime", "redis", "pydantic_settings")


def _module_check(name: str, *, required: bool) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"required": required, "available": False}
    try:
        spec = importlib.util.find_spec(name)
    except ValueError as exc:
        try:
            module = importlib.import_module(name)
        except Exception:
            result.update(
                {
                    "status": "missing_required" if required else "missing_optional",
                    "error": f"Python module {name!r} is unavailable: {exc}",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            return result
        result.update(
            {
                "status": "ok",
                "available": True,
                "version": getattr(module, "__version__", None),
                "origin": getattr(getattr(module, "__spec__", None), "origin", None),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return result
    if spec is None:
        result.update(
            {
                "status": "missing_required" if required else "missing_optional",
                "error": f"Python module {name!r} is not installed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return result

    try:
        module = importlib.import_module(name)
        result.update(
            {
                "status": "ok",
                "available": True,
                "version": getattr(module, "__version__", None),
                "origin": getattr(spec, "origin", None),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "import_failed_required" if required else "import_failed_optional",
                "error": f"{type(exc).__name__}: {exc}",
                "origin": getattr(spec, "origin", None),
            }
        )
    result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def _torch_runtime_details(torch_check: dict[str, Any]) -> dict[str, Any]:
    if not torch_check.get("available"):
        return {}
    try:
        torch = importlib.import_module("torch")
        mps_backend = getattr(torch.backends, "mps", None)
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "mps_built": bool(mps_backend and mps_backend.is_built()),
            "mps_available": bool(mps_backend and mps_backend.is_available()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def readiness_payload() -> dict[str, Any]:
    """Return a fast, non-mutating readiness payload without loading MiDaS weights."""
    started = time.perf_counter()
    required = {name: _module_check(name, required=True) for name in REQUIRED_RUNTIME_MODULES}
    optional = {name: _module_check(name, required=False) for name in OPTIONAL_RUNTIME_MODULES}
    required_ok = all(item.get("available") for item in required.values())
    torch_details = _torch_runtime_details(required.get("torch", {}))
    onnx_status = onnx_status_payload(settings.DEPTHLENS_WARMUP_DEVICE)
    asset_status = inspect_model_assets()
    asset_status["runtime_imports_ready"] = required_ok
    inference_ready = bool(required_ok and asset_status.get("model_assets_ready"))

    return {
        "status": "ready" if inference_ready else ("assets_missing" if required_ok else "degraded"),
        "backend_alive": True,
        "runtime_imports_ready": required_ok,
        "model_assets_ready": bool(asset_status.get("model_assets_ready")),
        "pytorch_hub_cache_ready": bool(asset_status.get("pytorch_hub_cache_ready")),
        "pytorch_hub_cache_path": asset_status.get("pytorch_hub_cache_path"),
        "midas_repo_cached": bool(asset_status.get("midas_repo_cached")),
        "checkpoint_summary": asset_status.get("checkpoint_summary"),
        "onnx_any_ready": bool(asset_status.get("onnx_any_ready")),
        "onnx_all_ready": bool(asset_status.get("onnx_all_ready")),
        "downloads_disabled": bool(asset_status.get("downloads_disabled")),
        "inference_ready": inference_ready,
        "fatal_reason": (
            None
            if inference_ready
            else (asset_status.get("fatal_reason") or "runtime_imports_unavailable")
        ),
        "recommended_action": (
            None
            if inference_ready
            else (
                asset_status.get("recommended_action")
                or "Install backend dependencies and run setup."
            )
        ),
        "required": required,
        "optional": optional,
        "torch_runtime": torch_details,
        "onnx_weights": {
            model: {
                "path": item["expected_path"],
                "exists": item["exists"],
                "size_bytes": item["size_bytes"],
            }
            for model, item in onnx_status["models"].items()
        },
        "onnx": onnx_status,
        "models": [{"id": model_id, **meta} for model_id, meta in SUPPORTED_MODELS.items()],
        "colormaps": list(COLORMAP_NAMES),
        "settings": {
            "host": settings.HOST,
            "port": settings.PORT,
            "max_dim": settings.DEPTHLENS_MAX_DIM,
            "preload_model": settings.DEPTHLENS_PRELOAD_MODEL,
            "warmup_model": settings.DEPTHLENS_WARMUP_MODEL,
            "warmup_device": settings.DEPTHLENS_WARMUP_DEVICE,
            "default_metrics": settings.DEPTHLENS_DEFAULT_METRICS,
            "default_outputs": settings.DEPTHLENS_DEFAULT_OUTPUTS,
        },
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
