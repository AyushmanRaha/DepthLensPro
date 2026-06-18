"""Depth-estimation model wrappers for PyTorch and ONNX Runtime execution."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch

from backend.model_registry import (
    get_model_spec,
    normalize_model_id,
    onnx_output_path,
    resolve_onnx_path,
)
from backend.utils.hardware import _default_device_key

MIDAS_REPO = "intel-isl/MiDaS"
log = logging.getLogger("depthlens")


def _torch_hub_load(*args: Any, **kwargs: Any) -> Any:
    hub_load = cast(Callable[..., Any], torch.hub.load)
    return hub_load(*args, **kwargs)


ONNX_INPUT_SIZE = (256, 256)
DEFAULT_ONNX_DIR = Path(__file__).resolve().parent / "onnx_weights"


def onnx_model_path(model_name: str, output_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the absolute expected static ONNX weight path for a supported model."""

    return onnx_output_path(model_name, output_dir=output_dir)


def resize_onnx_depth(pred: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize ONNX depth maps with PyTorch-compatible bicubic semantics where possible.

    OpenCV and PyTorch bicubic kernels are not bit-identical. ONNX uses OpenCV
    here to avoid a torch dependency in the runtime resize path, then clamps to
    the source depth range to reduce bicubic ringing at hard depth edges.
    """

    if pred.shape == output_shape:
        return pred.astype(np.float32, copy=False)
    import cv2

    lo = float(np.nanmin(pred))
    hi = float(np.nanmax(pred))
    interpolation = int(getattr(cv2, "INTER_CUBIC", 2))
    resized = cast(
        np.ndarray,
        cv2.resize(pred, (output_shape[1], output_shape[0]), interpolation=interpolation),
    )
    clipped = cast(np.ndarray, np.clip(resized, lo, hi).astype(np.float32, copy=False))
    return clipped


class ONNXExecutionEngine:
    """Run exported MiDaS ONNX graphs through the best available Runtime provider."""

    def __init__(
        self,
        model_name: str = "MiDaS_small",
        model_path: str | os.PathLike[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.model_id = normalize_model_id(model_name)
        self.spec = get_model_spec(self.model_id)
        self.model_name = self.spec.pytorch_model_name
        if model_path is None:
            resolved = resolve_onnx_path(self.model_id)
            path_value = resolved.get("onnx_path")
        else:
            self.model_path = Path(model_path).expanduser().resolve()
            exists = self.model_path.is_file()
            resolved = {
                "model_id": self.model_id,
                "display_name": self.spec.display_name,
                "onnx_path": os.fspath(self.model_path),
                "expected_path": os.fspath(self.model_path),
                "exists": exists,
                "size_bytes": self.model_path.stat().st_size if exists else None,
                "source": "explicit",
                "error": None if exists else "missing_file",
            }
            path_value = resolved["onnx_path"]

        if not path_value:
            raise FileNotFoundError(
                f"ONNX weights are unavailable for {self.spec.display_name}: "
                f"{resolved.get('expected_path') or '<unresolved>'} "
                f"({resolved.get('error') or 'missing_file'})."
            )
        self.model_path = Path(str(path_value)).expanduser().resolve()
        if not self.model_path.is_file() or self.model_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"ONNX weights are unavailable for {self.spec.display_name}: "
                f"{self.model_path} ({resolved.get('error') or 'missing_file'})."
            )

        ort = importlib.import_module("onnxruntime")
        self.session_options = self._session_options(ort)
        from backend.services.onnx_diagnostics import create_onnx_session

        session_result = create_onnx_session(
            self.model_id, device, model_path=self.model_path, session_options=self.session_options
        )
        if not session_result.get("ok"):
            log.warning(
                "ONNX_SESSION_CREATE_FAILED model=%s device=%s provider_state=%s "
                "providers=%s detail=%s",
                self.model_id,
                device,
                session_result.get("provider_state"),
                session_result.get("providers_used") or session_result.get("available_providers"),
                session_result.get("technical_detail") or session_result.get("message"),
            )
            raise RuntimeError(session_result.get("message") or "ONNX Runtime session unavailable")
        self.available_providers = list(session_result.get("available_providers", []))
        self.providers = list(session_result.get("providers_used", []))
        self.session = session_result["session"]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.transform = self._load_transform(model_name)

    @staticmethod
    def _load_transform(model_name: str) -> Any:
        transforms = _torch_hub_load(MIDAS_REPO, "transforms", trust_repo=True)
        model_id = normalize_model_id(model_name)
        if model_id == "midas_small":
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
        try:
            outputs = self.session.run([self.output_name], {self.input_name: batch})
        except Exception as exc:
            log.warning(
                "ONNX_SESSION_RUN_FAILED model=%s provider=%s input_shape=%s error=%s",
                self.model_id,
                self.provider,
                tuple(batch.shape),
                exc,
            )
            raise RuntimeError("ONNX Runtime inference failed") from exc
        pred = np.asarray(outputs[0], dtype=np.float32)
        pred = np.squeeze(pred)
        return resize_onnx_depth(pred, img_rgb.shape[:2])

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """Compatibility alias for callers expecting a model-like predict method."""

        return self.forward(img_rgb)


class DepthEstimator:
    """Legacy direct-use helper retained for external compatibility.

    The FastAPI/service inference path uses module-level loaders in
    :mod:`backend.services.inference` instead.  Keep new backend work out of this
    class so model loading and cache behavior do not diverge again.
    """

    __legacy__ = True

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
            spec = get_model_spec(model_name)
            self.models[model_name] = _torch_hub_load(
                self.repo, spec.pytorch_model_name, trust_repo=True
            )
            self.models[model_name].to(self.device)
            self.models[model_name].eval()

            midas_transforms = _torch_hub_load(self.repo, "transforms", trust_repo=True)
            if spec.model_id == "midas_small":
                self.transforms[model_name] = midas_transforms.small_transform
            else:
                self.transforms[model_name] = midas_transforms.dpt_transform

        return self.models[model_name], self.transforms[model_name]

    def load_onnx_engine(self, model_name: str) -> ONNXExecutionEngine:
        if model_name not in self.onnx_engines:
            self.onnx_engines[model_name] = ONNXExecutionEngine(model_name, device=str(self.device))
        return self.onnx_engines[model_name]

    def predict(self, img_rgb: np.ndarray, model_name: str) -> np.ndarray:
        if self.prefer_onnx and onnx_model_path(model_name).is_file():
            try:
                return self.load_onnx_engine(model_name).predict(img_rgb)
            except Exception:
                # Optional ONNX acceleration must never break the PyTorch path.
                pass

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
