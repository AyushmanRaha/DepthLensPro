"""Local TorchVision object detection for webcam capture previews."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

from PIL import Image, UnidentifiedImageError

MODEL_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"
MAX_INTERNAL_DIM = 960

COCO_LABELS = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
GENERIC_LABELS = {label for label in COCO_LABELS if label != "__background__"}

_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


class DetectorUnavailableError(RuntimeError):
    """Raised when local detector dependencies or weights are unavailable."""


def _resolve_device(requested: str) -> str:
    try:
        from backend.utils.hardware import _resolve

        return str(_resolve(requested))
    except ValueError:
        raise
    except Exception:
        return "cpu"


def get_detector(device: str = "auto") -> tuple[Any, str, Any]:
    """Load and cache the TorchVision detector once per resolved device."""

    try:
        import torch
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    except Exception as exc:  # pragma: no cover - depends on optional runtime wheels
        raise DetectorUnavailableError("TorchVision detector dependencies are unavailable") from exc

    resolved = _resolve_device(device)
    if resolved.startswith("mps"):
        # Some torchvision detection ops are still unreliable on MPS; prefer safe local CPU.
        resolved = "cpu"
    torch_device = torch.device(resolved)
    key = str(torch_device)
    with _model_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached, key, torch
        try:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            model.to(torch_device).eval()
        except Exception as exc:
            raise DetectorUnavailableError("Local object detector model is unavailable") from exc
        _model_cache[key] = model
        return model, key, torch


def _load_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a readable image") from exc
    if max(image.size) > MAX_INTERNAL_DIM:
        image.thumbnail((MAX_INTERNAL_DIM, MAX_INTERNAL_DIM), Image.Resampling.LANCZOS)
    return image


def detect_objects(
    raw: bytes,
    filename: str | None = None,
    device: str = "auto",
    threshold: float = 0.35,
    max_detections: int = 5,
) -> dict[str, Any]:
    """Run local COCO object detection and return generic labels only."""

    del filename
    started = time.perf_counter()
    image = _load_image(raw)
    width, height = image.size
    model, device_used, torch = get_detector(device)
    try:
        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(device_used)
        with torch.inference_mode():
            output = model([tensor])[0]
    except Exception as exc:
        if device_used != "cpu":
            model, device_used, torch = get_detector("cpu")
            from torchvision.transforms.functional import to_tensor

            tensor = to_tensor(image).to("cpu")
            with torch.inference_mode():
                output = model([tensor])[0]
        else:
            raise DetectorUnavailableError("Local object detector inference failed") from exc

    detections: list[dict[str, Any]] = []
    scores = output.get("scores", [])
    labels = output.get("labels", [])
    boxes = output.get("boxes", [])
    for label_id, score, box in zip(labels, scores, boxes, strict=False):
        score_f = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
        if score_f < threshold:
            continue
        idx = int(label_id.detach().cpu().item() if hasattr(label_id, "detach") else label_id)
        label = COCO_LABELS[idx] if 0 <= idx < len(COCO_LABELS) else "object"
        if label not in GENERIC_LABELS:
            continue
        coords = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
        detections.append(
            {
                "label": label,
                "score": round(score_f, 4),
                "box": [round(float(v), 2) for v in coords],
            }
        )
        if len(detections) >= max_detections:
            break

    detections.sort(key=lambda item: item["score"], reverse=True)
    return {
        "detections": detections,
        "model": MODEL_NAME,
        "device_used": device_used,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "resolution": {"width": width, "height": height},
    }


def detector_status(device: str = "auto") -> dict[str, Any]:
    """Return actionable diagnostics for the local detector without requiring inference."""

    status: dict[str, Any] = {
        "available": False,
        "model": MODEL_NAME,
        "device_requested": device,
        "device_selected": None,
        "torch": {"import_ok": False},
        "torchvision": {"import_ok": False},
        "weights": {"present": False, "source": "torchvision_default_cache"},
        "last_error": None,
    }
    try:
        import torch  # type: ignore

        status["torch"] = {"import_ok": True, "version": getattr(torch, "__version__", None)}
    except Exception as exc:
        status["last_error"] = f"torch import failed: {type(exc).__name__}: {exc}"
        return status
    try:
        import torchvision  # type: ignore
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

        status["torchvision"] = {
            "import_ok": True,
            "version": getattr(torchvision, "__version__", None),
        }
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        status["weights"].update(
            {"present": True, "name": str(weights), "source": "torchvision_default"}
        )
    except Exception as exc:
        status["last_error"] = f"torchvision/weights unavailable: {type(exc).__name__}: {exc}"
        return status
    try:
        resolved = _resolve_device(device)
        if resolved.startswith("mps"):
            status["warning"] = (
                "TorchVision detection uses CPU on Apple Silicon because MPS detection ops are unreliable."
            )
            resolved = "cpu"
        status["device_selected"] = resolved
        status["available"] = True
        return status
    except Exception as exc:
        status["last_error"] = f"device unavailable: {type(exc).__name__}: {exc}"
        return status
