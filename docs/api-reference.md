# API Reference

[← Back to README](../README.md)

Base URL:

```
http://127.0.0.1:8765
```

### Endpoint Overview

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service name and API version |
| `GET` | `/live` | Lightweight liveness check — fast, no dependencies |
| `GET` | `/ready` | Runtime dependency readiness — checks importability |
| `GET` | `/health` | Full diagnostics: devices, cache, ONNX, memory, disk |
| `GET` | `/devices` | Available compute devices |
| `GET` | `/models` | Supported model registry (canonical IDs, specs, input sizes) |
| `GET` | `/colormaps` | Supported colormap names |
| `GET` | `/onnx/status` | ONNX weight paths, provider availability, checker state |
| `GET` | `/benchmark` | PyTorch vs ONNX benchmark |
| `GET` | `/api/benchmark` | Frontend-compatible benchmark alias |
| `GET` | `/cache/metrics` | Cache telemetry (hits, misses, keyspace size, backend type) |
| `GET` | `/metrics` | Prometheus metrics exposition for local scraping |
| `GET` | `/api/observability` | JSON observability snapshot for the Performance panel |
| `GET` | `/observability` | Observability snapshot alias |
| `DELETE` | `/cache` | Clear all cache entries |
| `POST` | `/cache/clear` | Browser/client-friendly cache clear alias |
| `POST` | `/estimate` | Single-image depth estimation |
| `POST` | `/compare` | Multi-model comparison on one image |
| `POST` | `/api/compare` | Compare endpoint alias |
| `POST` | `/batch` | Batch depth estimation (up to 10 images) |
| `POST` | `/detect` | Local object detection for RGB Camera / 3D workflows |
| `POST` | `/api/detect` | Detection endpoint alias |
| `POST` | `/api/reconstruct` | 3D point-cloud reconstruction |
| `POST` | `/reconstruct` | Reconstruction alias |

---

### `POST /estimate`

Generates a depth map for one image.

#### Form Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | **required** | Input image, max 20 MB |
| `model` | string | `MiDaS_small` | `MiDaS_small`, `DPT_Hybrid`, or `DPT_Large` (aliases normalised) |
| `colormap` | string | `inferno` | Any supported colormap name |
| `device` | string | `auto` | `auto`, `cpu`, `mps`, `cuda:0`, `xpu:0`, etc. |
| `metrics` | string | `fast` | `none`, `fast`, or `full` |
| `outputs` | string | `color` | `color`, `gray`, or `color,gray` |
| `max_dim` | integer | `1536` via config | Resize long edge before inference |
| `gt_file` | file | optional | PNG/TIFF/NPY ground-truth depth file |
| `gt_required` | boolean | `false` | Return 422 if GT file is missing |
| `gt_scale` | float | optional | Multiplier applied to GT values (e.g. `0.001` for mm→m) |
| `gt_invalid_value` | float | optional | Sentinel value to mask from GT before metrics |

#### Example

```bash
curl -X POST http://127.0.0.1:8765/estimate \
  -F "file=@photo.jpg" \
  -F "model=MiDaS_small" \
  -F "colormap=inferno" \
  -F "device=auto" \
  -F "metrics=fast" \
  -F "outputs=color"
```

#### Response Fields

| Field | Description |
|---|---|
| `depth_map` | Base64 PNG colourised depth map |
| `grayscale` | Base64 PNG greyscale depth map (when `outputs` includes `gray`) |
| `metrics` | Grouped prediction stats, proxy metrics, and optional GT metrics |
| `latency_ms` | Server-side forward-pass time in milliseconds |
| `model_id` | Canonical model ID |
| `device_used` | Resolved runtime device string |
| `engine_used` | `pytorch`, `onnxruntime`, or `cache` |
| `fallback_used` | `true` when ONNX was requested but PyTorch was used instead |
| `cached` | `true` when response came from cache |
| `resolution` | `{"width": W, "height": H}` of the processed image |
| `gt_metadata` | GT processing details, scale, alignment, valid pixel counts |

---


### `POST /compare`

Runs multiple supported depth models on one image and returns per-model outputs plus a compact comparison summary. `/api/compare` is available as a frontend-compatible alias.

#### Form Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | **required** | Input image, max 20 MB |
| `models` | string | all supported comparison models | Comma-separated model IDs or aliases, deduplicated after normalisation |
| `colormap` | string | `inferno` | Any supported colormap name |
| `device` | string | `auto` | `auto`, `cpu`, `mps`, `cuda:0`, `xpu:0`, etc. |
| `metrics` | string | `full` | `none`, `fast`, or `full` |
| `outputs` | string | `color,gray` | `color`, `gray`, or `color,gray` |
| `max_dim` | integer | optional | Resize long edge before inference |

#### Example

```bash
curl -X POST http://127.0.0.1:8765/compare \
  -F "file=@photo.jpg" \
  -F "models=MiDaS_small,DPT_Hybrid,DPT_Large" \
  -F "colormap=inferno" \
  -F "device=auto" \
  -F "metrics=full" \
  -F "outputs=color,gray"
```

#### Response Shape

```json
{
  "filename": "photo.jpg",
  "device_used": "cpu",
  "models": ["midas_small", "dpt_hybrid", "dpt_large"],
  "results": [
    {
      "model_id": "midas_small",
      "depth_map": "...",
      "grayscale": "...",
      "metrics": {},
      "latency_ms": 123.45,
      "engine_used": "pytorch",
      "fallback_used": false,
      "cached": false,
      "resolution": {"width": 640, "height": 480}
    }
  ],
  "errors": [],
  "total": 3,
  "succeeded": 3,
  "failed": 0,
  "comparison": {
    "fastest_model_id": "midas_small",
    "lowest_latency_ms": 123.45,
    "slowest_model_id": "dpt_large",
    "highest_latency_ms": 456.78
  }
}
```

Each model is validated, cached, and processed independently. Runtime failures for one model are returned in `errors` while the endpoint continues with the remaining requested models where safe.

---

### `POST /batch`

Runs depth estimation on multiple images concurrently (up to 10).

Each file is independently validated, cached, and processed. Non-image files and files over 20 MB are reported as errors without stopping the remaining items.

```bash
curl -X POST http://127.0.0.1:8765/batch \
  -F "files=@image_1.jpg" \
  -F "files=@image_2.jpg" \
  -F "model=MiDaS_small" \
  -F "colormap=inferno" \
  -F "device=auto"
```

Response shape:

```json
{
  "results": [...],
  "errors": [{"filename": "bad.txt", "error": "Image file required"}],
  "total": 2,
  "succeeded": 1,
  "failed": 1
}
```

---

### `POST /detect` and `POST /api/detect`

Runs local TorchVision COCO object detection for the RGB Camera / 3D workflow using `fasterrcnn_mobilenet_v3_large_320_fpn`. It is an RGB preview/capture aid, not a depth model. Detector weights and optional dependencies may be unavailable on a machine; in that case the endpoint returns a structured detector-unavailable error with remediation details instead of silently falling back.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | **required** | Input image; max upload limit follows the backend upload limit and detector processing caps the internal long edge at 960 px |
| `device` | string | `auto` | Runtime device selection |
| `threshold` | float | `0.35` | Detection confidence threshold; valid range `0.05` to `0.95` |
| `max_detections` | integer | `5` | Maximum detections to return; valid range `1` to `20` |

The response includes `detections`, `model`, `device_used`, `latency_ms`, and the processed `resolution`. MPS requests are resolved to CPU for this detector, and failed non-CPU inference is retried once on CPU.

---

### Cache endpoints

- `GET /cache/metrics` returns active cache telemetry, including hit/miss counters, keyspace size, and backend type.
- `DELETE /cache` clears the active inference cache.
- `POST /cache/clear` also clears the active inference cache for browser/client flows that prefer POST requests.

---

### `POST /api/reconstruct`

Generates an approximate point cloud from a source image.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | **required** | Source image, max 20 MB |
| `model` | string | `MiDaS_small` | Depth model |
| `device` | string | `auto` | Runtime device |
| `colormap` | string | `inferno` | Depth visualisation colormap |
| `max_dim` | integer | optional | Resize before depth inference |
| `export_format` | string | `ply` | `ply` or `obj` |
| `max_points` | integer | `120000` | Export point budget |
| `preview_points` | integer | `5000` | In-app WebGL preview budget |
| `focal_scale` | float | `1.2` | Approximate focal length multiplier |
| `depth_scale` | float | `1.0` | Z-axis multiplier |
| `depth_near_percentile` | float | `2.0` | Near clipping percentile (clips foreground outliers) |
| `depth_far_percentile` | float | `98.0` | Far clipping percentile (clips background outliers) |
| `sampling` | string | `grid` | `grid`, `stride`, or `random` (seed 0) |
| `include_rgb` | boolean | `true` | Embed source-image pixel colours per point |
| `coordinate_system` | string | `y_up` | `y_up` (Y negated) or `z_up` (raw projection). Invalid values return a structured validation error. |

---

### `GET /benchmark`

Benchmarks PyTorch and ONNX using a synthetic 384×384 frame.

```bash
curl "http://127.0.0.1:8765/benchmark?model=MiDaS_small&device=auto&iterations=3"
```

Query parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | `MiDaS_small` | Model to benchmark |
| `device` | `auto` | Runtime device |
| `iterations` | `3` | Number of timing iterations (clamped to 1–20) |

The benchmark runs under a global mutex that prevents concurrent benchmark calls, and sets a `/live` response field `"state": "busy"` so the frontend can indicate that a benchmark is in progress.

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---


## Error response schema

Route-level failures return FastAPI-compatible envelopes under `detail`:

```json
{
  "detail": {
    "error_code": "INVALID_CONTENT_TYPE",
    "message": "Expected an image file",
    "field": "file",
    "retryable": false
  }
}
```

Optional fields include `remediation`, `field`, and `retryable`. Batch and compare keep their existing per-item/per-model success flow and legacy `error` strings where present, while also adding `error_detail` with the same structured fields. Timeout codes are route-specific: `/benchmark` returns `BENCHMARK_TIMEOUT`, `/reconstruct` returns `RECONSTRUCTION_TIMEOUT`, and `/estimate`, `/compare`, `/batch`, and `/detect` return `REQUEST_TIMEOUT`.

### Adaptive engine mode
Depth routes (`/estimate`, `/batch`, `/compare`, `/api/reconstruct`, and aliases) accept optional `engine` form values: `auto` (default), `pytorch`, or `onnxruntime` (`onnx` alias). Responses add `engine_requested`, `engine_used`, `engine_selection`, `fallback_used`, and `fallback_reason` while preserving existing fields. `/benchmark` accepts query `engine=both|auto|pytorch|onnxruntime` (default `both`) and returns warmup-aware timing fields such as `latency_ms.samples`, `first_run_ms`/`cold_start_ms`, `warmup_iterations`, plus `comparison.recommended_engine`, `comparison.recommendation_reason`, and `comparison.display_label`.

### `GET /api/detect/status`

Returns local camera detector readiness without requiring an image upload. By default this is a lightweight status check and does not load the detector. Pass `?warmup=true` or call `POST /api/detect/warmup` to perform a single-flight detector warmup. Response fields include `available`, `state`, `device_requested`, `device_used`, `model`, `message`, `last_error`, and `warmup_in_progress`.

### Liveness versus readiness

`GET /live` is process liveness only and does not import inference runtimes or scan model assets. Use `/ready`, `/health`, and `/onnx/status` for deeper diagnostics. ONNX status distinguishes an asset being present from a verified runtime session; forced ONNX Runtime attempts the preferred provider first, then a CPUExecutionProvider ONNX fallback before using PyTorch fallback.


### Liveness, readiness, and activity

`GET /live` is import-light process liveness. It returns `busy`, `state`, `active_operations`, `total_active_operations`, timestamps, PID, service, version, and uptime. It does not import Torch, OpenCV, NumPy, ONNX Runtime, inference services, detector services, or model assets. `GET /ready` reports runtime readiness, and `GET /health` reports diagnostics that may be stale or degraded without meaning the engine is dead.

Compare responses include `results[]`, `errors[]`, `succeeded`, `failed`, and `total`. Per-model errors include the model id/display name, `error_code`, user message, technical detail, requested engine/device, and elapsed time when available.

Inference engine values are `auto`, `pytorch`, `onnxruntime_prefer`, and `onnxruntime_strict`; the legacy `onnxruntime` value remains accepted as prefer-mode compatibility. ONNX execution failures report stage-specific diagnostics such as `onnx_failure_stage=execution`, provider names, inputs/outputs, input shape/dtype, exception type, and root exception message.

`/api/detect/status` uses `ready` to mean the detector model is loaded. `/api/detect` reports `detections`, `detection_count`, `threshold`, `max_detections`, `model_loaded`, `device_used`, and `latency_ms`; an empty list is a successful “no object detected” result.
