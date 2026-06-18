"""Central filesystem path policy for DepthLens Pro resources."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
MODEL_ROOT = REPO_ROOT / "models"
TORCH_CACHE_ROOT = MODEL_ROOT / "torch-cache"
ONNX_ROOT = MODEL_ROOT / "onnx"
LEGACY_ONNX_ROOT = BACKEND_ROOT / "onnx_weights"


def user_cache_onnx_root() -> Path:
    """Return the per-user fallback ONNX cache directory."""

    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "DepthLensPro" / "onnx"


def torch_cache_root() -> Path:
    """Return the active Torch cache root, honoring TORCH_HOME."""

    return Path(os.getenv("TORCH_HOME", TORCH_CACHE_ROOT)).expanduser().resolve()


def onnx_root() -> Path:
    """Return the canonical repo ONNX root without legacy fallback lookup."""

    return ONNX_ROOT.resolve()


def onnx_candidate_dirs(output_dir: str | os.PathLike[str] | None = None) -> list[tuple[str, Path]]:
    """Return ONNX search/write directories in existing priority order."""

    if output_dir:
        return [("explicit", Path(output_dir).expanduser())]
    if os.getenv("DEPTHLENSPRO_MODEL_DIR"):
        base = Path(os.environ["DEPTHLENSPRO_MODEL_DIR"]).expanduser()
        return [("env_model_dir", base / "onnx"), ("env_model_dir_legacy", base)]
    if os.getenv("DEPTHLENS_ONNX_DIR"):
        return [("env_onnx_dir", Path(os.environ["DEPTHLENS_ONNX_DIR"]).expanduser())]
    if os.getenv("ONNX_WEIGHTS_DIR"):
        return [("env_weights_dir", Path(os.environ["ONNX_WEIGHTS_DIR"]).expanduser())]
    return [("repo", ONNX_ROOT), ("legacy", LEGACY_ONNX_ROOT), ("cache", user_cache_onnx_root())]
