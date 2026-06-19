# How Monocular Depth Estimation Works

[← Back to README](../README.md)

Understanding what happens inside the pipeline helps you choose the right model, interpret the metrics, and tune output quality.

### 1. Image Ingestion and Preprocessing

The uploaded image is decoded by OpenCV into a BGR NumPy array. If the longest edge exceeds `DEPTHLENS_MAX_DIM` (default 1536 px), the image is down-scaled with area interpolation to keep inference fast without sacrificing perceptual quality.

Each MiDaS model family applies its own preprocessing transform (loaded from Torch Hub's `transforms` module):

- **MiDaS Small** — `small_transform`: resizes to 256×256, normalises with ImageNet statistics, and produces a `(1, 3, 256, 256)` float32 tensor.
- **DPT Hybrid / DPT Large** — `dpt_transform`: resizes to 384×384 with padding that preserves aspect ratio, then normalises. The resulting tensor is `(1, 3, 384, 384)`.

### 2. Forward Pass (PyTorch or ONNX Runtime)

**PyTorch path:**

```
input tensor → model.forward() → (1, H_out, W_out) raw depth tensor
```

The model is loaded once via `torch.hub.load("intel-isl/MiDaS", ...)`, moved to the selected device, and set to `model.eval()`. Forward passes run inside `torch.inference_mode()` to skip gradient bookkeeping. A per-model lock prevents concurrent forward calls on the same model/device instance, which can be unsafe on some backends.

After the forward pass, the output is bicubic-upsampled back to the original image resolution using `torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)`.

**ONNX path:**

The exported ONNX graph takes the same preprocessed tensor as input. The session is created with `onnxruntime.InferenceSession`, selecting providers in priority order: CUDA > CoreML > OpenVINO > CPU. ORT's graph optimizer runs at `ORT_ENABLE_ALL`, and both intra-op and inter-op parallelism are configured via environment variables. After inference, the output depth map is bicubic-resized back to the source resolution using OpenCV (with clamping to prevent ringing artefacts at depth edges).

### 3. Depth Normalisation

Raw model output is an unbounded float32 plane where larger values mean *farther* (MiDaS convention). The inference service normalises this to `[0, 1]`:

```
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
```

This makes the output comparable across different images and models, but also means metric values (metres) are lost. If metric depth is needed, pair the model output with camera calibration and a reference measurement.

### 4. Colourisation and Encoding

The normalised depth plane is multiplied by 255, cast to `uint8`, and passed through an OpenCV colour map (`cv2.applyColorMap`). The resulting BGR image is PNG-encoded and base64-serialised for the HTTP response. Greyscale output follows the same path but converts via `cv2.COLOR_GRAY2BGR`.

### 5. Inference Cache

The full `(model, colormap, device, metrics_mode, outputs, max_dim, image_content_hash)` tuple forms the cache key (SHA-1 of the serialised parameters). If Redis is configured, results are stored as versioned JSON with a configurable TTL (default 3600 s). If Redis is unavailable, the service falls back to a thread-safe in-process LRU dict capped at `CACHE_MAX_ENTRIES` entries. GT depth uploads bypass the cache entirely to prevent stale payloads contaminating benchmark runs.

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---
