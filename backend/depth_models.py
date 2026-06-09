"""Depth-estimation model wrappers for PyTorch and ONNX Runtime execution."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, cast

import cv2  # noqa: F401 - retained for compatibility with legacy imports.
import numpy as np
import torch

from backend.utils.hardware import _default_device_key, _onnx_providers_for_device

MIDAS_REPO = "intel-isl/MiDaS"
ONNX_INPUT_SIZE = (384, 384)
DEFAULT_ONNX_DIR = Path(__file__).resolve().parent / "onnx_weights"


def onnx_model_path(model_name: str, output_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the expected static ONNX weight path for a MiDaS model."""

    configured_dir = output_dir or os.getenv("DEPTHLENS_ONNX_DIR")
    base_dir = Path(configured_dir) if configured_dir is not None else DEFAULT_ONNX_DIR
    return base_dir / f"{model_name}.onnx"


class ONNXExecutionEngine:
    """Run exported MiDaS ONNX graphs through the best available Runtime provider."""

    def __init__(
        self,
        model_name: str = "MiDaS_small",
        model_path: str | os.PathLike[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else onnx_model_path(model_name)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX weights not found for {model_name!r}: {self.model_path}. "
                "Run backend/scripts/export_onnx.py first."
            )

        ort = importlib.import_module("onnxruntime")
        self.available_providers = list(ort.get_available_providers())
        self.providers = _onnx_providers_for_device(device, self.available_providers)
        self.session_options = self._session_options(ort)
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=self.session_options, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.transform = self._load_transform(model_name)

    @staticmethod
    def _load_transform(model_name: str) -> Any:
        transforms = torch.hub.load(  # type: ignore[no-untyped-call]
            MIDAS_REPO, "transforms", trust_repo=True
        )
        if model_name == "MiDaS_small":
            return transforms.small_transform
        return transforms.dpt_transform

    @staticmethod
    def _session_options(ort: Any) -> Any:
        """Create optimized ONNX Runtime session options for CV inference."""

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        default_intra_threads = min(4, os.cpu_count() or 1)
        intra_threads = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", str(default_intra_threads)))
        inter_threads = int(os.getenv("ORT_INTER_OP_NUM_THREADS", "1"))
        options.intra_op_num_threads = max(1, intra_threads)
        options.inter_op_num_threads = max(1, inter_threads)
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        return options

    @property
    def provider(self) -> str:
        """Return the active first-choice ONNX Runtime provider."""

        return self.providers[0] if self.providers else "unknown"

    def forward(self, img_rgb: np.ndarray) -> np.ndarray:
        """Execute an ONNX Runtime forward pass and return the raw depth plane."""

        batch = self.transform(img_rgb).detach().cpu().numpy().astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        pred = np.asarray(outputs[0], dtype=np.float32)
        pred = np.squeeze(pred)
        if pred.shape != img_rgb.shape[:2]:
            pred = cast(
                np.ndarray,
                cv2.resize(
                    pred, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC
                ),
            )
        return pred.astype(np.float32, copy=False)

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """Compatibility alias for callers expecting a model-like predict method."""

        return self.forward(img_rgb)


class DepthEstimator:
    """Small legacy helper for callers that use depth_models.py directly."""

    def __init__(self, prefer_onnx: bool = True) -> None:
        self.device = torch.device(_default_device_key())
        self.models: dict[str, torch.nn.Module] = {}
        self.transforms: dict[str, Any] = {}
        self.onnx_engines: dict[str, ONNXExecutionEngine] = {}
        self.repo = MIDAS_REPO
        self.prefer_onnx = prefer_onnx

    def load_model(self, model_name: str) -> tuple[torch.nn.Module, Any]:
        if model_name not in self.models:
            print(f"Loading {model_name} onto {self.device}...")
            self.models[model_name] = torch.hub.load(  # type: ignore[no-untyped-call]
                self.repo, model_name, trust_repo=True
            )
            self.models[model_name].to(self.device)
            self.models[model_name].eval()

            midas_transforms = torch.hub.load(  # type: ignore[no-untyped-call]
                self.repo, "transforms", trust_repo=True
            )
            if model_name == "MiDaS_small":
                self.transforms[model_name] = midas_transforms.small_transform
            else:
                self.transforms[model_name] = midas_transforms.dpt_transform

        return self.models[model_name], self.transforms[model_name]

    def load_onnx_engine(self, model_name: str) -> ONNXExecutionEngine:
        if model_name not in self.onnx_engines:
            self.onnx_engines[model_name] = ONNXExecutionEngine(model_name, device=str(self.device))
        return self.onnx_engines[model_name]

    def predict(self, img_rgb: np.ndarray, model_name: str) -> np.ndarray:
        if self.prefer_onnx and onnx_model_path(model_name).exists():
            return self.load_onnx_engine(model_name).predict(img_rgb)

        model, transform = self.load_model(model_name)
        input_batch = transform(img_rgb).to(self.device)

        with torch.inference_mode():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.detach().cpu().numpy()
