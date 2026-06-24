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
| `POST` | `/estimate` | Single-image depth estimation |
| `POST` | `/compare` | Multi-model comparison on one image |
| `POST` | `/api/compare` | Compare endpoint alias |
| `POST` | `/batch` | Batch depth estimation (up to 10 images) |
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
| `sampling` | string | `grid` | `grid` (deterministic) or `random` (seed 0) |
| `include_rgb` | boolean | `true` | Embed source-image pixel colours per point |
| `coordinate_system` | string | `y_up` | `y_up` (Y negated) or `camera` (raw projection) |

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
