"""PyTorch MiDaS model loading, transform caching, and inference helpers."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, cast

import cv2
import numpy as np
import torch

from backend.model_registry import get_model_spec, normalize_model_id
from backend.services.image_io import _normalize_depth
from backend.services.model_assets import (
    ModelAssetsUnavailableError,
    classify_torch_hub_failure,
    ensure_assets_before_load,
    inspect_model_assets,
)

log = logging.getLogger("depthlens")

MODELS: dict[str, tuple[torch.nn.Module, torch.device]] = {}
TRANSFORMS: dict[str, Callable[[np.ndarray], torch.Tensor]] = {}
_MODEL_LOCK = threading.RLock()
_MODEL_FORWARD_LOCKS: dict[str, threading.Lock] = {}


def _torch_hub_load(*args: Any, **kwargs: Any) -> Any:
    hub_load = cast(Callable[..., Any], torch.hub.load)
    return hub_load(*args, **kwargs)


def _load_model(
    model_name: str, device_str: str
) -> tuple[tuple[torch.nn.Module, torch.device], Callable[[np.ndarray], torch.Tensor]]:
    """Load or reuse a MiDaS model for a resolved device."""
    model_id = normalize_model_id(model_name)
    spec = get_model_spec(model_id)
    key = f"{model_id}:{device_str}"
    with _MODEL_LOCK:
        if key in MODELS:
            return MODELS[key], TRANSFORMS[model_id]

        ensure_assets_before_load()
        if model_id not in TRANSFORMS:
            try:
                transforms = _torch_hub_load("intel-isl/MiDaS", "transforms", trust_repo=True)
            except Exception as exc:
                log.exception(
                    "MiDaS transforms load failed; TORCH_HOME=%s", os.getenv("TORCH_HOME")
                )
                if classify_torch_hub_failure(exc):
                    raise ModelAssetsUnavailableError(
                        status=inspect_model_assets(), cause=exc
                    ) from exc
                raise
            TRANSFORMS[model_id] = (
                transforms.small_transform
                if model_id == "midas_small"
                else transforms.dpt_transform
            )

        device = torch.device(device_str)
        log.info("Loading '%s' (%s) → %s …", spec.display_name, spec.pytorch_model_name, device)
        try:
            model = _torch_hub_load("intel-isl/MiDaS", spec.pytorch_model_name, trust_repo=True)
        except Exception as exc:
            log.exception(
                "MiDaS model load failed for %s; TORCH_HOME=%s",
                spec.pytorch_model_name,
                os.getenv("TORCH_HOME"),
            )
            if classify_torch_hub_failure(exc):
                raise ModelAssetsUnavailableError(status=inspect_model_assets(), cause=exc) from exc
            raise
        model.to(device).eval()

        if device.type == "mps":
            model = model.float()

        MODELS[key] = (model, device)
        _MODEL_FORWARD_LOCKS.setdefault(key, threading.Lock())
        log.info("✅ '%s' ready on %s", spec.display_name, device)
        return (model, device), TRANSFORMS[model_id]


def _infer_torch(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    """Run normalized PyTorch depth inference for benchmarking and fallback."""

    model_id = normalize_model_id(model_name)
    (model, device), transform = _load_model(model_id, device_str)
    key = f"{model_id}:{device_str}"
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device, non_blocking=True)
    with torch.inference_mode():
        with _MODEL_FORWARD_LOCKS[key]:
            pred = model(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    del batch, pred
    return _normalize_depth(depth)
