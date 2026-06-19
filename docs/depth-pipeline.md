# Depth Pipeline

DepthLens Pro uses pretrained MiDaS/DPT models to predict **relative depth**. It does not estimate calibrated real-world distance.

## 1. Image ingestion

The API receives uploaded image bytes, validates request options, decodes the image with Pillow/OpenCV-compatible paths, and prepares RGB data for inference.

## 2. Preprocessing

Images may be resized to fit the selected maximum dimension. The pipeline tracks original dimensions so model output can be resized back for visualization, metrics, and export.

## 3. PyTorch path

The standard path uses PyTorch and MiDaS/DPT assets from the local Torch Hub cache. Models are selected by canonical ID, loaded lazily, and protected by per-model/device forward locks.

## 4. ONNX path

The optional ONNX path uses ONNX Runtime when local `.onnx` files exist under `models/onnx` and provider diagnostics pass. ONNX is optional for standard builds.

## 5. Normalization

Raw model output is normalized into a visualization-friendly depth plane. Because monocular depth is relative, values are meaningful for near/far ordering inside the same prediction, not absolute measurement.

## 6. Colorization

The normalized depth plane is mapped through the selected colormap and encoded for app preview/export.

## 7. Cache-key behavior

Inference cache keys include image content plus model, device, colormap, requested outputs, metrics mode, and sizing options. Ground-truth requests bypass image-only cache reuse so evaluation reflects the current GT file.
