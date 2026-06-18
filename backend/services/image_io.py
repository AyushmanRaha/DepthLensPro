"""Image decode, normalization, colorization, and encoding helpers."""

from __future__ import annotations

import base64
from typing import cast

import cv2
import numpy as np

from backend.config import settings
from backend.model_metadata import COLORMAP_NAMES

COLORMAPS: dict[str, int] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    "turbo": cv2.COLORMAP_TURBO,
}
assert set(COLORMAPS) == set(COLORMAP_NAMES)

MAX_DIM = int(settings.DEPTHLENS_MAX_DIM)


def _decode(raw: bytes, max_dim: int | None = None) -> np.ndarray:
    """Decode image bytes into a BGR OpenCV image, resizing oversized inputs."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — corrupt or unsupported format")
    h, w = img.shape[:2]
    limit = max(256, int(max_dim or MAX_DIM))
    if max(h, w) > limit:
        scale = limit / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a raw depth plane into the existing [0, 1] output range."""

    depth = depth.astype(np.float32, copy=False)
    lo, hi = depth.min(), depth.max()
    return cast(np.ndarray, (depth - lo) / (hi - lo + 1e-8))


def _colorize(depth: np.ndarray, cmap: str) -> np.ndarray:
    """Apply an OpenCV color map to a normalized depth map."""
    u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, COLORMAPS.get(cmap, cv2.COLORMAP_INFERNO))


def _b64(img: np.ndarray) -> str:
    """Encode an image as a PNG base64 string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()
