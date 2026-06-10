"""Static model and colormap metadata that is safe to import without ML runtimes."""

from __future__ import annotations

from backend.model_registry import MODEL_REGISTRY

SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    model_id: {
        "label": spec.display_name,
        "display_name": spec.display_name,
        "description": spec.notes or spec.architecture,
        "compute": spec.recommended_device,
        "pytorch_model_name": spec.pytorch_model_name,
    }
    for model_id, spec in MODEL_REGISTRY.items()
}

COLORMAP_NAMES: tuple[str, ...] = (
    "inferno",
    "plasma",
    "viridis",
    "magma",
    "jet",
    "hot",
    "bone",
    "turbo",
)
