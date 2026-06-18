"""Model asset discovery and actionable errors for packaged/offline MiDaS runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from backend.model_registry import DEFAULT_MODEL_DIR, MODEL_REGISTRY, resolve_onnx_path

MIDAS_MODEL_IDS = tuple(MODEL_REGISTRY.keys())
MODEL_TO_CHECKPOINT_HINTS = {
    "midas_small": ("midas_v21_small", "midas_small"),
    "dpt_hybrid": ("dpt_hybrid", "dpt_hybrid_384"),
    "dpt_large": ("dpt_large", "dpt_large_384"),
}


class CheckpointFile(TypedDict):
    name: str
    size_bytes: int


class ModelAssetsUnavailableError(RuntimeError):
    """Raised when MiDaS assets are missing in offline/packaged runtime."""

    error_code = "MODEL_ASSETS_UNAVAILABLE"

    def __init__(
        self,
        message: str | None = None,
        *,
        status: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message or "MiDaS model assets are missing or incomplete.")
        self.status = status or inspect_model_assets()
        self.__cause__ = cause

    def to_payload(self) -> dict[str, Any]:
        return model_assets_error_payload(self.status)


def downloads_disabled() -> bool:
    return os.getenv("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def torch_home() -> Path:
    raw = os.getenv("TORCH_HOME")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_MODEL_DIR / "torch-cache"


def _checkpoint_summary(cache: Path) -> dict[str, Any]:
    checkpoints = cache / "hub" / "checkpoints"
    files: list[CheckpointFile] = []
    if checkpoints.is_dir():
        for path in sorted(checkpoints.iterdir()):
            if path.is_file() and path.suffix.lower() in {".pt", ".pth"}:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                files.append({"name": path.name, "size_bytes": size})
    by_model = {}
    for model_id, hints in MODEL_TO_CHECKPOINT_HINTS.items():
        matches = [
            file
            for file in files
            if file["size_bytes"] > 0 and any(hint in file["name"].lower() for hint in hints)
        ]
        by_model[model_id] = {"ready": bool(matches), "files": matches}
    return {
        "path": os.fspath(checkpoints),
        "exists": checkpoints.is_dir(),
        "files": files,
        "by_model": by_model,
    }


def _midas_repo(cache: Path) -> dict[str, Any]:
    hub = cache / "hub"
    candidates = []
    if hub.is_dir():
        candidates = [p for p in hub.iterdir() if p.is_dir() and "midas" in p.name.lower()]
    valid = []
    for repo in candidates:
        if (repo / "hubconf.py").is_file() and (
            (repo / "midas").is_dir() or (repo / "MiDaS").is_dir()
        ):
            valid.append(repo)
    return {
        "cached": bool(valid),
        "candidates": [os.fspath(p) for p in candidates],
        "valid_repos": [os.fspath(p) for p in valid],
    }


def onnx_asset_status() -> dict[str, Any]:
    models = {mid: resolve_onnx_path(mid) for mid in MIDAS_MODEL_IDS}
    any_ready = any(
        item.get("exists") and int(item.get("size_bytes") or 0) > 0 for item in models.values()
    )
    all_ready = all(
        item.get("exists") and int(item.get("size_bytes") or 0) > 0 for item in models.values()
    )
    return {"models": models, "onnx_any_ready": any_ready, "onnx_all_ready": all_ready}


def inspect_model_assets(
    *, required_models: tuple[str, ...] = MIDAS_MODEL_IDS, cache_root: Path | None = None
) -> dict[str, Any]:
    cache = cache_root or torch_home()
    repo = _midas_repo(cache)
    checkpoints = _checkpoint_summary(cache)
    required_ok = all(checkpoints["by_model"].get(mid, {}).get("ready") for mid in required_models)
    pytorch_ready = cache.is_dir() and repo["cached"] and required_ok
    onnx = onnx_asset_status()
    recommended = (
        None
        if pytorch_ready
        else (
            "Run npm run setup:<platform> and rebuild the app. "
            "For ONNX builds run npm run setup:<platform>:onnx."
        )
    )
    return {
        "runtime_imports_ready": None,
        "model_assets_ready": pytorch_ready,
        "pytorch_hub_cache_ready": pytorch_ready,
        "pytorch_hub_cache_path": os.fspath(cache),
        "midas_repo_cached": repo["cached"],
        "midas_repo": repo,
        "checkpoint_summary": checkpoints,
        "onnx_any_ready": onnx["onnx_any_ready"],
        "onnx_all_ready": onnx["onnx_all_ready"],
        "onnx": onnx,
        "downloads_disabled": downloads_disabled(),
        "inference_ready": pytorch_ready,
        "fatal_reason": None if pytorch_ready else "pytorch_midas_cache_missing_or_incomplete",
        "recommended_action": recommended,
    }


def model_assets_error_payload(status: dict[str, Any] | None = None) -> dict[str, Any]:
    status = status or inspect_model_assets()
    return {
        "error_code": "MODEL_ASSETS_UNAVAILABLE",
        "message": "MiDaS model assets are missing or incomplete.",
        "action": status.get("recommended_action")
        or "Run npm run setup:<platform> and rebuild the app.",
        "torch_home": status.get("pytorch_hub_cache_path"),
        "expected_cache": status.get("pytorch_hub_cache_path"),
        "standard_build_note": (
            "ONNX is optional; PyTorch MiDaS assets are required for standard builds."
        ),
        "details": status,
    }


def classify_torch_hub_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return downloads_disabled() or any(
        token in text
        for token in (
            "no such file",
            "not found",
            "unable to find",
            "urlopen",
            "connection",
            "permission",
            "read-only",
            "readonly",
            "checkpoints",
        )
    )


def ensure_assets_before_load() -> None:
    if downloads_disabled():
        status = inspect_model_assets()
        if not status["pytorch_hub_cache_ready"]:
            raise ModelAssetsUnavailableError(status=status)
