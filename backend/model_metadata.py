"""Static model and colormap metadata that is safe to import without ML runtimes."""

from __future__ import annotations

SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "MiDaS_small": {
        "label": "Small",
        "description": "~30 ms · EfficientNet-Lite · CPU-friendly",
        "compute": "CPU or GPU",
    },
    "DPT_Hybrid": {
        "label": "Hybrid",
        "description": "~120 ms · ViT-Hybrid · GPU recommended",
        "compute": "GPU recommended",
    },
    "DPT_Large": {
        "label": "Large",
        "description": "~400 ms · ViT-Large · GPU required for speed",
        "compute": "GPU required",
    },
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
