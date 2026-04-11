import logging

import cv2
import numpy as np
import torch

from .config import SUPPORTED_MODELS
from .devices import resolve_device
from .runtime import MODELS, TRANSFORMS

logger = logging.getLogger("depthlens")


def load_model(model_name: str, device_str: str):
    key = f"{model_name}:{device_str}"
    if key in MODELS:
        return MODELS[key], TRANSFORMS[model_name]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")

    device = resolve_device(device_str)
    logger.info("Loading '%s' -> %s …", model_name, device)
    model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    model.to(device).eval()

    if device.type == "mps":
        model = model.float()

    if model_name not in TRANSFORMS:
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        TRANSFORMS[model_name] = (
            midas_transforms.small_transform
            if model_name == "MiDaS_small"
            else midas_transforms.dpt_transform
        )

    MODELS[key] = (model, device)
    logger.info("✅ '%s' ready on %s", model_name, device)
    return (model, device), TRANSFORMS[model_name]


def infer_depth(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    (model, device), transform = load_model(model_name, device_str)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)

    with torch.no_grad():
        pred = model(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = pred.cpu().numpy().astype(np.float32)
    low, high = depth.min(), depth.max()
    return (depth - low) / (high - low + 1e-8)
