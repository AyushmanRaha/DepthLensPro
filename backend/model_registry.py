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
        input_size=(256, 256),
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


def supported_models_legacy_payload() -> dict[str, dict[str, str]]:
    """Return the legacy lightweight metadata shape used by older callers."""

    return {
        model_id: {
            "label": spec.display_name,
            "display_name": spec.display_name,
            "description": spec.notes or spec.architecture,
            "compute": spec.recommended_device,
            "pytorch_model_name": spec.pytorch_model_name,
        }
        for model_id, spec in MODEL_REGISTRY.items()
    }


def _candidate_dirs(output_dir: str | os.PathLike[str] | None = None) -> list[tuple[str, Path]]:
    """Return deterministic ONNX search/write directories in priority order."""

    if output_dir:
        return [("explicit", Path(output_dir).expanduser())]
    if os.getenv("DEPTHLENSPRO_MODEL_DIR"):
        base = Path(os.environ["DEPTHLENSPRO_MODEL_DIR"]).expanduser()
        return [("env_model_dir", base / "onnx"), ("env_model_dir_legacy", base)]
    if os.getenv("DEPTHLENS_ONNX_DIR"):
        return [("env_onnx_dir", Path(os.environ["DEPTHLENS_ONNX_DIR"]).expanduser())]
    if os.getenv("ONNX_WEIGHTS_DIR"):
        return [("env_weights_dir", Path(os.environ["ONNX_WEIGHTS_DIR"]).expanduser())]
    return [("repo", DEFAULT_ONNX_DIR), ("legacy", LEGACY_ONNX_DIR), ("cache", USER_CACHE_ONNX_DIR)]


def _path_payload(
    spec: ModelSpec,
    *,
    selected_path: Path | None,
    selected_source: str | None,
    candidates: list[tuple[str, Path]],
    error: str | None,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    exists = bool(selected_path and selected_path.is_file())
    size = selected_path.stat().st_size if selected_path and exists else None
    reported_expected_path = expected_path or selected_path
    return {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "onnx_path": os.fspath(selected_path) if selected_path else None,
        "expected_path": os.fspath(reported_expected_path) if reported_expected_path else None,
        "exists": exists,
        "size_bytes": size,
        "source": selected_source,
        "error": (
            error
            if error is not None
            else ("empty_file" if exists and int(size or 0) <= 0 else None)
        ),
        "candidates": [os.fspath(path) for _, path in candidates],
    }


def resolve_onnx_path(
    model: str | None,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    for_write: bool = False,
) -> dict[str, Any]:
    """Resolve ONNX paths deterministically for reads or writes.

    Read mode returns only an existing file in ``onnx_path``. Missing files are
    reported with ``onnx_path=None`` and the canonical destination in
    ``expected_path`` so callers cannot accidentally treat a guessed path as a
    valid model. Write mode always returns the canonical write destination.
    """

    try:
        spec = get_model_spec(model)
    except UnknownModelError as exc:
        return {
            "model_id": model or "",
            "display_name": None,
            "onnx_path": None,
            "expected_path": None,
            "exists": False,
            "size_bytes": None,
            "source": None,
            "error": "unknown_model",
            "error_code": exc.error_code,
            "valid_models": exc.valid_models,
            "candidates": [],
        }

    dirs = _candidate_dirs(output_dir)
    candidates = [
        (source, (directory / spec.onnx_filename).resolve()) for source, directory in dirs
    ]
    if not candidates:
        return _path_payload(
            spec,
            selected_path=None,
            selected_source=None,
            candidates=[],
            error="empty_path",
        )

    canonical_source, canonical_path = candidates[0]
    if for_write:
        return _path_payload(
            spec,
            selected_path=canonical_path,
            selected_source=canonical_source,
            candidates=candidates,
            error=None if canonical_path.is_file() else "missing_file",
            expected_path=canonical_path,
        )

    for source, path in candidates:
        if path.is_file():
            error = "empty_file" if path.stat().st_size <= 0 else None
            return _path_payload(
                spec,
                selected_path=path,
                selected_source=source,
                candidates=candidates,
                error=error,
                expected_path=canonical_path,
            )

    return _path_payload(
        spec,
        selected_path=None,
        selected_source=None,
        candidates=candidates,
        error="missing_file",
        expected_path=canonical_path,
    )


def onnx_output_path(model: str | None, output_dir: str | os.PathLike[str] | None = None) -> Path:
    resolved = resolve_onnx_path(model, output_dir=output_dir, for_write=True)
    path = resolved.get("onnx_path")
    if not path:
        raise UnknownModelError(model or "")
    return Path(path)
