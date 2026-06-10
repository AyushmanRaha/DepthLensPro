"""Central model registry, name normalization, and asset path resolution."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class UnknownModelError(ValueError):
    """Raised when a model identifier cannot be normalized to a supported model."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Unknown model id: {model_id}")
        self.model_id = model_id
        self.error_code = "UNKNOWN_MODEL"
        self.valid_models = list(MODEL_REGISTRY)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    architecture: str
    input_size: tuple[int, int]
    pytorch_model_name: str
    preprocessing: str
    pytorch_supported: bool
    onnx_supported: bool
    onnx_filename: str
    recommended_device: str
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_size"] = list(self.input_size)
        return payload


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_ONNX_DIR = DEFAULT_MODEL_DIR / "onnx"
LEGACY_ONNX_DIR = Path(__file__).resolve().parent / "onnx_weights"
USER_CACHE_ONNX_DIR = (
    Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "DepthLensPro" / "onnx"
)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "midas_small": ModelSpec(
        model_id="midas_small",
        display_name="MiDaS Small",
        architecture="MiDaS small / EfficientNet-Lite",
        input_size=(384, 384),
        pytorch_model_name="MiDaS_small",
        preprocessing="small_transform",
        pytorch_supported=True,
        onnx_supported=True,
        onnx_filename="midas_small.onnx",
        recommended_device="cpu_or_gpu",
        notes="Fastest supported MiDaS model; PyTorch fallback is always allowed.",
    ),
    "dpt_hybrid": ModelSpec(
        model_id="dpt_hybrid",
        display_name="DPT Hybrid",
        architecture="DPT Hybrid / ViT-Hybrid",
        input_size=(384, 384),
        pytorch_model_name="DPT_Hybrid",
        preprocessing="dpt_transform",
        pytorch_supported=True,
        onnx_supported=True,
        onnx_filename="dpt_hybrid.onnx",
        recommended_device="gpu_preferred",
        notes="ONNX export may require legacy static-shape export on some PyTorch versions.",
    ),
    "dpt_large": ModelSpec(
        model_id="dpt_large",
        display_name="DPT Large",
        architecture="DPT Large / ViT-Large",
        input_size=(384, 384),
        pytorch_model_name="DPT_Large",
        preprocessing="dpt_transform",
        pytorch_supported=True,
        onnx_supported=True,
        onnx_filename="dpt_large.onnx",
        recommended_device="gpu_required_for_speed",
        notes="Largest supported model; ONNX export is optional and may fail on some runtimes.",
    ),
}

_ALIASES: dict[str, str] = {}
for canonical, spec in MODEL_REGISTRY.items():
    variants = {
        canonical,
        canonical.replace("_", " "),
        canonical.replace("_", "-"),
        spec.display_name,
        spec.display_name.replace(" ", "_"),
        spec.display_name.replace(" ", "-"),
        spec.pytorch_model_name,
        spec.pytorch_model_name.replace("_", " "),
        spec.pytorch_model_name.replace("_", "-"),
    }
    for variant in variants:
        _ALIASES[variant.lower().replace("-", "_").replace(" ", "_")] = canonical


def normalize_model_id(value: str | None) -> str:
    """Return the canonical model id for known UI/backend naming variants."""

    key = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _ALIASES.get(key)
    if not canonical:
        raise UnknownModelError(value or "")
    return canonical


def get_model_spec(value: str | None) -> ModelSpec:
    return MODEL_REGISTRY[normalize_model_id(value)]


def supported_models_payload() -> list[dict[str, Any]]:
    return [spec.as_dict() for spec in MODEL_REGISTRY.values()]


def _candidate_dirs(output_dir: str | os.PathLike[str] | None = None) -> list[tuple[str, Path]]:
    if output_dir:
        return [("explicit", Path(output_dir).expanduser())]
    if os.getenv("DEPTHLENSPRO_MODEL_DIR"):
        base = Path(os.environ["DEPTHLENSPRO_MODEL_DIR"]).expanduser()
        return [("env", base / "onnx"), ("env", base)]
    if os.getenv("DEPTHLENS_ONNX_DIR"):
        return [("env", Path(os.environ["DEPTHLENS_ONNX_DIR"]).expanduser())]
    return [("repo", DEFAULT_ONNX_DIR), ("legacy", LEGACY_ONNX_DIR), ("cache", USER_CACHE_ONNX_DIR)]


def resolve_onnx_path(
    model: str | None,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    for_write: bool = False,
) -> dict[str, Any]:
    """Resolve an absolute ONNX path and report existence without returning empty paths."""

    try:
        spec = get_model_spec(model)
    except UnknownModelError as exc:
        return {
            "model_id": model or "",
            "display_name": None,
            "onnx_path": None,
            "exists": False,
            "size_bytes": None,
            "source": None,
            "error": "unknown_model",
            "error_code": exc.error_code,
            "valid_models": exc.valid_models,
        }

    dirs = _candidate_dirs(output_dir)
    candidates = [
        (source, (directory / spec.onnx_filename).resolve()) for source, directory in dirs
    ]
    if not candidates:
        return {
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "onnx_path": None,
            "exists": False,
            "size_bytes": None,
            "source": None,
            "error": "empty_path",
        }

    selected_source, selected_path = candidates[0]
    for source, path in candidates:
        if path.exists():
            selected_source, selected_path = source, path
            break
    if for_write and output_dir is None and not any(path.exists() for _, path in candidates):
        selected_source, selected_path = candidates[0]

    path_str = os.fspath(selected_path)
    exists = selected_path.is_file()
    size = selected_path.stat().st_size if exists else None
    error = None
    if not path_str:
        error = "empty_path"
    elif not exists:
        error = "missing_file"
    elif not size or size <= 0:
        error = "empty_file"

    return {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "onnx_path": path_str,
        "exists": exists,
        "size_bytes": size,
        "source": selected_source,
        "error": error,
        "candidates": [os.fspath(path) for _, path in candidates],
    }


def onnx_output_path(model: str | None, output_dir: str | os.PathLike[str] | None = None) -> Path:
    resolved = resolve_onnx_path(model, output_dir=output_dir, for_write=True)
    path = resolved.get("onnx_path")
    if not path:
        raise UnknownModelError(model or "")
    return Path(path)
