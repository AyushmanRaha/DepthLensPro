"""Lightweight model asset preflight checks for DepthLens Pro."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from backend.model_registry import MODEL_REGISTRY, REPO_ROOT, resolve_onnx_path

ERROR_CODE = "MODEL_ASSETS_UNAVAILABLE"
MESSAGE = "MiDaS model assets are missing. Run setup to cache PyTorch MiDaS assets, or install ONNX weights for acceleration."
ACTION = "Run npm run setup, then npm run verify:model-assets. If network access is blocked, copy a pre-cached models/torch-cache directory into this repo."
STANDARD_BUILD_NOTE = "ONNX is optional; PyTorch MiDaS assets are required when ONNX weights are absent."
MIDAS_REPO_MARKERS = ("hubconf.py", "midas", "transforms.py")


class ModelAssetsUnavailableError(RuntimeError):
    """Raised when neither optional ONNX nor PyTorch MiDaS assets can run inference."""

    error_code = ERROR_CODE
    message = MESSAGE
    action = ACTION
    standard_build_note = STANDARD_BUILD_NOTE

    def __init__(self, message: str | None = None, *, cause: BaseException | None = None) -> None:
        super().__init__(message or self.message)
        self.__cause__ = cause

    def to_response(self) -> dict[str, str]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "action": self.action,
            "standard_build_note": self.standard_build_note,
        }


def model_downloads_disabled() -> bool:
    value = os.getenv("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def torch_home() -> Path:
    return Path(os.getenv("TORCH_HOME", REPO_ROOT / "models" / "torch-cache")).expanduser()


def _midas_repo_candidates(cache_root: Path | None = None) -> list[Path]:
    root = cache_root or torch_home()
    hub = root / "hub"
    candidates = [hub / "intel-isl_MiDaS_master"]
    if hub.is_dir():
        candidates.extend(sorted(p for p in hub.glob("intel-isl_MiDaS*") if p.is_dir()))
    return list(dict.fromkeys(candidates))


def _repo_cached(cache_root: Path | None = None) -> tuple[bool, Path | None]:
    for candidate in _midas_repo_candidates(cache_root):
        if (candidate / "hubconf.py").is_file() and ((candidate / "midas").is_dir() or (candidate / "transforms.py").is_file()):
            return True, candidate
    return False, None


def _check_import(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"available": spec is not None, "origin": getattr(spec, "origin", None) if spec else None}


def model_assets_status(*, deep: bool = False) -> dict[str, Any]:
    imports = {name: _check_import(name) for name in ("torch", "torchvision", "cv2", "PIL", "numpy")}
    required_imports_ready = all(imports[name]["available"] for name in ("torch", "cv2", "PIL", "numpy"))
    onnx_models = {model_id: resolve_onnx_path(model_id) for model_id in MODEL_REGISTRY}
    any_onnx_ready = any(bool(item.get("exists")) and int(item.get("size_bytes") or 0) > 0 for item in onnx_models.values())
    all_onnx_ready = all(bool(item.get("exists")) and int(item.get("size_bytes") or 0) > 0 for item in onnx_models.values())
    hub_root = torch_home()
    repo_ok, repo_path = _repo_cached(hub_root)
    downloads_disabled = model_downloads_disabled()
    # A cached hub repo is the cheap, offline signal that torch.hub can import transforms/model code.
    pytorch_assets_ready = repo_ok
    can_run_offline = any_onnx_ready or pytorch_assets_ready
    model_assets_ready = any_onnx_ready or pytorch_assets_ready
    inference_ready = required_imports_ready and model_assets_ready
    standard_build_ready = required_imports_ready and (pytorch_assets_ready or any_onnx_ready or not downloads_disabled)
    fatal_reason = None
    if required_imports_ready and not model_assets_ready and downloads_disabled:
        fatal_reason = "ONNX weights are absent and the PyTorch MiDaS torch hub cache is missing while model downloads are disabled."
    elif not required_imports_ready:
        fatal_reason = "One or more required runtime imports are unavailable."
    status = "ready" if inference_ready else ("unavailable" if fatal_reason else "degraded")
    return {
        "status": status,
        "standard_build_ready": bool(standard_build_ready),
        "runtime_imports_ready": bool(required_imports_ready),
        "model_assets_ready": bool(model_assets_ready),
        "inference_ready": bool(inference_ready),
        "imports": imports,
        "python_executable": sys.executable,
        "onnx": {"any_ready": any_onnx_ready, "all_ready": all_onnx_ready, "models": onnx_models, "optional": True},
        "pytorch_hub": {
            "torch_home": os.fspath(hub_root),
            "hub_dir": os.fspath(hub_root / "hub"),
            "midas_repo_cached": repo_ok,
            "midas_repo_path": os.fspath(repo_path) if repo_path else None,
            "candidate_paths": [os.fspath(p) for p in _midas_repo_candidates(hub_root)],
            "downloads_disabled": downloads_disabled,
            "assets_likely_available": pytorch_assets_ready,
        },
        "can_run_inference_offline": bool(can_run_offline),
        "fatal_reason": fatal_reason,
        "recommended_action": None if inference_ready else ACTION,
        "standard_build_note": STANDARD_BUILD_NOTE,
        "deep": deep,
    }


def should_treat_hub_error_as_assets_unavailable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = ("github", "internet", "network", "connection", "urlopen", "http", "not found", "no such file", "errno", "midas")
    return model_downloads_disabled() or any(marker in text for marker in markers)
