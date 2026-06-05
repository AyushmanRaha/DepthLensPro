"""Legacy DepthEstimator wrapper around MiDaS models."""

from __future__ import annotations

from typing import Any

import cv2  # noqa: F401 - retained for compatibility with legacy imports.
import numpy as np
import torch

from backend.utils.hardware import _default_device_key


class DepthEstimator:
    """Small legacy helper for callers that use depth_models.py directly."""

    def __init__(self) -> None:
        self.device = torch.device(_default_device_key())
        self.models: dict[str, torch.nn.Module] = {}
        self.transforms: dict[str, Any] = {}
        self.repo = "intel-isl/MiDaS"

    def load_model(self, model_name: str) -> tuple[torch.nn.Module, Any]:
        if model_name not in self.models:
            print(f"Loading {model_name} onto {self.device}...")
            self.models[model_name] = torch.hub.load(  # type: ignore[no-untyped-call]
                self.repo, model_name
            )
            self.models[model_name].to(self.device)
            self.models[model_name].eval()

            midas_transforms = torch.hub.load(  # type: ignore[no-untyped-call]
                self.repo, "transforms"
            )
            if model_name == "MiDaS_small":
                self.transforms[model_name] = midas_transforms.small_transform
            else:
                self.transforms[model_name] = midas_transforms.dpt_transform

        return self.models[model_name], self.transforms[model_name]

    def predict(self, img_rgb: np.ndarray, model_name: str) -> np.ndarray:
        model, transform = self.load_model(model_name)
        input_batch = transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()
