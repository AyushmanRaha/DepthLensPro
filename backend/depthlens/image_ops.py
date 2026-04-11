import base64
import hashlib

import cv2
import numpy as np

from .config import COLORMAPS, MAX_DIM


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — corrupt or unsupported format")

    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def colorize_depth(depth: np.ndarray, cmap: str) -> np.ndarray:
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, COLORMAPS.get(cmap, cv2.COLORMAP_INFERNO))


def encode_png_b64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf.tobytes()).decode()


def make_cache_key(raw: bytes, model: str, cmap: str, device: str) -> str:
    payload = f"{model}:{cmap}:{device}:{hashlib.md5(raw).hexdigest()}"
    return hashlib.sha1(payload.encode()).hexdigest()
