import cv2

APP_NAME = "DepthLens Pro API"
APP_VERSION = "3.0.0"
MAX_DIM = 2048
MAX_SIZE_MB = 20
MAX_BATCH_FILES = 10

SUPPORTED_MODELS = {
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

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    "turbo": cv2.COLORMAP_TURBO,
}
