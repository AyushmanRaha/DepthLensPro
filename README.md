<div align="center">

# DepthLens Pro

### Local-first monocular depth estimation for desktop workflows

Turn ordinary 2D images into depth maps, compare neural depth models, benchmark ONNX acceleration, evaluate against ground truth, and export approximate 3D point clouds — all from a desktop app running on your own machine.

<br/>

[![Desktop App](https://img.shields.io/badge/Desktop_App-v4.0.0-111827?style=for-the-badge)](electron-app/package.json)
[![API](https://img.shields.io/badge/API-v3.1.0-2563eb?style=for-the-badge)](backend/api/live.py)
[![Electron](https://img.shields.io/badge/Electron-42.3.0-47848f?style=for-the-badge&logo=electron&logoColor=white)](electron-app/package.json)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688?style=for-the-badge&logo=fastapi&logoColor=white)](backend/requirements.txt)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab?style=for-the-badge&logo=python&logoColor=white)](scripts/doctor.py)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](backend/requirements.txt)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.24.3-005ced?style=for-the-badge)](backend/requirements.txt)
[![Local First](https://img.shields.io/badge/Privacy-Local_First-16a34a?style=for-the-badge)](#security)
[![License: MIT](https://img.shields.io/badge/License-MIT-facc15?style=for-the-badge)](LICENSE)

<br/>

**No cloud uploads. No API keys. No subscription.**
Images are processed through a local Electron + FastAPI + PyTorch/ONNX pipeline.

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#installation-guide"><strong>Installation</strong></a> ·
  <a href="#api-reference"><strong>API Reference</strong></a> ·
  <a href="#troubleshooting"><strong>Troubleshooting</strong></a> ·
  <a href="#security"><strong>Security</strong></a> ·
  <a href="#contributing"><strong>Contributing</strong></a>
</p>

## Demo Video

### Quick Preview

<video src="https://github.com/user-attachments/assets/6baef599-df4c-4d48-9fa5-7cbe43a08449" controls muted playsinline width="100%">
  Your browser does not support the video tag.
</video>

### Full Demo

[Watch the full-quality 3:45 demo video](https://github.com/AyushmanRaha/DepthLensPro/releases/download/v1.0-demo/demo.mp4)

</div>

---

## Official v1.0 Tech Stack

| Component | Locked version / support | Source | Role |
|---|---:|---|---|
| Electron | `^42.3.0` | `electron-app/package.json` | Desktop shell, window lifecycle, preload bridge, packaged app builds |
| FastAPI | `0.135.3` | `backend/requirements.txt` | Local HTTP API framework |
| Python | `3.10–3.12` | `scripts/doctor.py`, setup checks | Backend runtime and ML environment |
| PyTorch | `2.11.0` | `backend/requirements.txt` | MiDaS/DPT model runtime through Torch Hub |
| ONNX Runtime | `1.24.3` | `backend/requirements.txt` | Optional accelerated inference and benchmark engine |
| Node.js / npm | Node LTS recommended / npm bundled with Node | setup diagnostics | Build tooling and Electron dependency installation |
| Redis | `6.4.0` Python client; Redis server optional | `backend/requirements.txt`, `docker-compose.yml` | Optional TTL cache backend |
| OpenCV | `opencv-python-headless==4.13.0.92` | `backend/requirements.txt` | Image decode, resizing, colorization, GT alignment |
| NumPy | `2.4.4` | `backend/requirements.txt` | Depth array arithmetic and metric computation |
| Pillow | `12.2.0` | `backend/requirements.txt` | PNG/TIFF image handling |
| Chart.js | `4.4.0` | `frontend/index.html` | Frontend latency, comparison, benchmark, and observability charts |

## Table of Contents

| Section | What it covers |
|---|---|
| [Official v1.0 Tech Stack](#official-v10-tech-stack) | Locked dependency versions verified from manifests |
| [Overview](#overview) | What the app does and who it is useful for |
| [What I built vs what I used](#what-i-built-vs-what-i-used) | Ownership boundaries between this application and integrated open-source components |
| [How Monocular Depth Estimation Works](#how-monocular-depth-estimation-works) | The ML pipeline explained step by step |
| [Highlights](#highlights) | Core capabilities at a glance |
| [Feature Tour](#feature-tour) | Workspace, webcam, comparison, experiments, performance, and 3D tools |
| [System Architecture](#system-architecture) | Electron, FastAPI, PyTorch, ONNX, and cache data flow |
| [System Design Decisions](#system-design-decisions) | Caching, concurrency, fallback, packaging, and local-first tradeoffs |
| [Quick Start](#quick-start) | Fast setup for local use |
| [Installation Guide](#installation-guide) | Native, development, backend-only, Docker, and ONNX setup |
| [Terminal-Only Dev Verification](#terminal-only-dev-verification) | Multi-OS dev-mode checks without packaging |
| [Configuration](#configuration) | Environment variables and runtime settings |
| [API Reference](#api-reference) | HTTP endpoints, request fields, and response behavior |
| [Models, Colormaps & Metrics](#models-colormaps--metrics) | Supported MiDaS/DPT models and evaluation modes |
| [Ground Truth Evaluation](#ground-truth-evaluation) | GT file support and benchmark metric flow |
| [Understanding Depth Metrics](#understanding-depth-metrics) | What each metric measures and when to use it |
| [Testing & CI](#testing--ci) | Local checks and GitHub Actions pipeline |
| [Production & Packaging](#production--packaging) | Platform-specific builds, ONNX variants, and Docker deployment |
| [Troubleshooting](#troubleshooting) | Common setup/runtime problems and fixes |
| [Security](#security) | Local-first design, renderer isolation, and process safeguards |
| [Project Structure](#project-structure) | Repository map |
| [Contributing](#contributing) | Development and PR checklist |
| [License](#license) | MIT License |
| [Acknowledgements](#acknowledgements) | Open-source projects powering DepthLens Pro |

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Overview

**DepthLens Pro** is a desktop application for generating **monocular depth maps** from regular images.

In plain English: give the app a photo, choose a model, and it predicts which parts of the scene are closer or farther away. The output can be used for visual effects, depth-aware editing, computer vision experiments, point-cloud previews, and model benchmarking.

Technically, the app combines:

- **Electron** for the desktop shell (main process, renderer, secure preload bridge)
- **FastAPI** for the local inference HTTP API (async routes, CORS, structured logging)
- **PyTorch Torch Hub** for MiDaS/DPT model execution (lazy-loaded, device-aware)
- **ONNX Runtime** for optional accelerated inference (CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, and more)
- **OpenCV / NumPy / Pillow** for image decoding, depth normalization, and GT processing
- **Redis or in-memory LRU cache** for repeated inference acceleration
- **Versioned Electron userData settings** with atomic writes, corruption recovery, and a browser localStorage fallback
- **Central platform/resource/ONNX resolvers** for supported native packages and diagnostics
- **Docker Compose** for containerized backend + Redis deployment

> DepthLens Pro predicts **relative depth**, not real-world metric distance. It is useful for visual depth understanding and approximate geometry, not survey-grade measurement.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## What I built vs what I used

DepthLens Pro is an application and systems engineering project built around **pretrained MiDaS/DPT models**. The repository does not claim ownership of MiDaS model research, model architecture, or pretrained weights. Instead, it turns those models and supporting runtimes into a local-first desktop product with a maintained inference API, UI workflow, diagnostics, packaging, benchmarking, evaluation, and export pipeline.

| Area | Built in this project | Used / integrated |
|---|---|---|
| ML model | Model selection, request validation, local orchestration, output normalization, colormap rendering, and response shaping | Pretrained MiDaS/DPT models from Intel ISL MiDaS through PyTorch Torch Hub |
| Desktop app | Electron shell, renderer workflow, secure preload bridge, backend lifecycle management, settings persistence, and export UX | Electron runtime and browser platform APIs |
| Backend inference API | FastAPI routes for `/estimate`, `/batch`, `/benchmark`, `/reconstruct`, health checks, diagnostics, and structured responses | FastAPI, Uvicorn, Pillow, OpenCV, NumPy, and runtime libraries |
| Caching and fallback | Cache-key design, Redis-to-memory fallback behavior, cache eligibility rules, and local telemetry integration | Redis as an optional external cache server; in-process memory for fallback |
| ONNX acceleration | ONNX path resolution, provider diagnostics, local validation, and optional runtime dispatch | ONNX Runtime and locally generated ONNX model files |
| Benchmarking/metrics | Benchmark endpoints, latency capture, metric presentation, and reviewer-facing diagnostics | Standard depth-estimation metric formulas and scientific Python tooling |
| Ground truth evaluation | GT upload handling, resizing/masking/alignment flow, benchmark integration, and result reporting | User-supplied GT files plus NumPy/OpenCV/Pillow processing |
| Point-cloud export | Depth-to-point-cloud workflow, preview downsampling, and PLY/OBJ export path | Standard pinhole-camera projection concepts and 3D file formats |
| Packaging/deployment | Desktop packaging configuration, platform/resource/ONNX resolvers, Docker backend workflow, and local-first startup checks | Electron Builder, Docker Compose, native OS packages, PyTorch, ONNX Runtime, and optional Redis |

In short, the project contribution is the productized local inference system: integration, orchestration, desktop UX, diagnostics, caching, benchmarking, evaluation, packaging, and export workflows around pretrained depth models.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## How Monocular Depth Estimation Works

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

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Highlights

| Capability | What it means |
|---|---|
| 🖼️ **Image-to-depth generation** | Upload one or more images and export colourised or greyscale depth maps |
| ⚙️ **Model selection** | Choose MiDaS Small, DPT Hybrid, or DPT Large depending on speed/quality tradeoff |
| 🧠 **Device selection** | Auto-detection or manual selection of CPU, CUDA GPU, Apple MPS/Neural Engine, or Intel XPU |
| 🎨 **Colormap control** | Visualise depth using Inferno, Plasma, Viridis, Magma, Turbo, Jet, Hot, or Bone |
| 📷 **Live webcam depth** | Process webcam frames locally at a controlled FPS with optional temporal smoothing |
| 📊 **Model comparison** | Run all supported models on the same image and compare outputs side-by-side |
| ⚡ **PyTorch vs ONNX benchmark** | Measure latency, throughput, memory, provider status, and speedup |
| 📡 **Local observability** | Prometheus metrics, inference telemetry, traces, crash analytics, and benchmark history in the Performance panel |
| 🧪 **Experiment exports** | Save validation runs as JSON or CSV for reproducible reporting |
| 📏 **Ground truth evaluation** | Compare predictions against PNG/TIFF/NPY depth labels with median-scale alignment |
| 🧊 **3D point clouds** | Export approximate coloured point clouds as PLY or OBJ |
| 🔒 **Local-first privacy** | Images are processed on `127.0.0.1`; no cloud inference is ever required |

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Feature Tour

### Observability — Local Runtime Telemetry

Observability lives inside the existing **Performance** panel as a second sub-view next to Benchmark. It exposes local-only runtime snapshots, Prometheus metrics, inference latency history, bounded trace spans, sanitized crash analytics, cache events, and benchmark history without adding a new top-level header tab.

The backend provides `GET /metrics` for Prometheus exposition plus `GET /api/observability` and `GET /observability` for JSON snapshots used by the UI. Telemetry is bounded in memory and avoids raw images, base64 image payloads, filenames, image hashes, cache keys, local paths, and high-cardinality user data in labels or history.

### 1. Workspace — Generate Depth Maps

<p align="center">
  <img src="docs/screenshots/workspace-generate-depth-maps.png" alt="Workspace — Generate Depth Maps" width="900">
</p>

The main workspace handles the complete image-processing flow:

1. Select a compute device (or leave on Auto to use the best detected accelerator).
2. Pick a model architecture.
3. Choose a depth colourmap.
4. Drop images into the queue.
5. Generate and download results.

Supported upload formats include **PNG, JPG, WEBP, and BMP**, up to **20 MB per image**. Images that exceed `max_dim` are automatically down-scaled before inference; the original file is never modified.

The workspace also includes a live session dashboard:

| Metric | Meaning |
|---|---|
| Images processed | Successful inference runs in the current session |
| Average / min / max latency | Server-side model forward-pass timing (excludes network and file I/O) |
| Cache hits | Repeated image/model/colormap combinations served without re-inference |
| Errors | Failed image-processing attempts |
| Throughput | Estimated images processed per minute, derived from average latency |
| Total inference time | Cumulative sum of all model forward-pass times |

---

### 2. Ground Truth Mode

<p align="center">
  <img src="docs/screenshots/ground-truth-mode.png" alt="Ground Truth Mode" width="900">
</p>

Ground Truth mode processes **one image and one depth label file** together, computing standard benchmark metrics with median-scale alignment.

Supported GT formats:

- `.png` — single-channel 8-bit or 16-bit preferred
- `.tif` / `.tiff` — single-channel floating-point or integer depth
- `.npy` — raw NumPy depth array (loaded with `allow_pickle=False` for safety)

Rules:

- GT file size limit: **20 MB**
- Multi-channel GT files are converted to greyscale via luminance weights (0.299 R + 0.587 G + 0.114 B)
- GT is resized to the prediction resolution using nearest-neighbour interpolation to avoid interpolating sentinel invalid values across boundaries
- An optional `gt_scale` multiplier converts between depth units (e.g. `0.001` to convert millimetres → metres)
- An optional `gt_invalid_value` masks sensor dropout pixels before metric computation
- Median-scale alignment is applied before any error metric is calculated

> **Why nearest-neighbour?** Linear or bicubic interpolation during GT resize would blend valid depth measurements with zero or invalid sentinel pixels, producing synthetic depth values at object boundaries. Nearest-neighbour sampling preserves the original label integrity.

---

### 3. Webcam — Live Depth Streaming

<p align="center">
  <img src="docs/screenshots/webcam-live-depth-streaming.png" alt="Webcam — Live Depth Streaming" width="900">
</p>

The Webcam tab processes a live camera feed into repeated depth predictions.

Controls:

| Control | Options |
|---|---|
| Start / stop camera | Requests camera permission and begins local capture |
| Pause inference | Keeps the camera active while halting model calls |
| Target FPS | 1, 2, 3, or 5 FPS (hard cap to protect the backend) |
| Frame max dimension | 256, 384, or 512 px (longer edge; aspect ratio is preserved) |
| Visual smoothing | Off, Low (α=0.25), Medium (α=0.45), or High (α=0.65) |
| Capture | Downloads the latest depth frame as PNG |

**Temporal smoothing** uses exponential moving average on raw pixel values between frames:

```
output_pixel = α × previous_pixel + (1 − α) × current_pixel
```

Higher α produces a smoother but more lag-prone result. Smoothing is automatically disabled when the page is hidden to avoid blending stale frames.

The live view shows real-time backend latency, end-to-end latency (including browser encoding), effective FPS, skipped frame count, and the currently active model, device, and colourmap.

---

### 4. Compare — Run All Models on One Image

<p align="center">
  <img src="docs/screenshots/compare-run-all-models.png" alt="Compare — Run All Models on One Image" width="900">
</p>

The Compare tab answers the practical question of which model is right for a scene:

> Should I use the fastest model, the balanced model, or the highest-detail model?

Upload one image, click **Run All Models**, and the app runs inference with:

- MiDaS Small (256×256 input, ~30 ms on CPU)
- DPT Hybrid (384×384 input, ~120 ms on GPU)
- DPT Large (384×384 input, ~400 ms on GPU)

The comparison view shows side-by-side depth previews, latency badges, and a switchable metric chart covering latency, SSIM, SILog, PSNR, gradient mean, edge density, entropy, and dynamic range.

---

### 5. Performance — PyTorch vs ONNX Runtime

<p align="center">
  <img src="docs/screenshots/performance-pytorch-vs-onnx.png" alt="Performance — PyTorch vs ONNX Runtime" width="900">
</p>

The Performance tab benchmarks the standard PyTorch path against optional ONNX Runtime execution using a synthetic 384×384 gradient frame (deterministic, no file upload needed).

Reported fields:

| Field | Description |
|---|---|
| PyTorch avg latency | Average time for `model.forward()` + bicubic resize, across N iterations |
| ONNX avg latency | Average time for `session.run()` + OpenCV bicubic resize, across N iterations |
| Speedup | `pytorch_avg / onnx_avg` — values >1 mean ONNX is faster |
| ONNX throughput | Synthetic frames per second at ONNX avg latency |
| Process RSS | Resident process memory (MB) measured after the benchmark completes |
| Execution status | Provider name (e.g. `CUDAExecutionProvider`) or fallback state with the expected ONNX path |

ONNX weights are **not committed** to this repository and must be generated locally. If ONNX files are absent or invalid, the benchmark reports their expected path, provides the exact export command, and continues running the PyTorch half of the test without interruption.

---

### 6. Experiments — Reproducible Validation Runs

<p align="center">
  <img src="docs/screenshots/experiments-validation-runs.png" alt="Experiments — Reproducible Validation Runs" width="900">
</p>

The Experiments tab records structured results from the current workspace queue into named runs.

You can:

- Name a run (default: `DepthLens validation run`)
- Execute all queued images
- Include optional ground-truth metrics when GT mode is enabled in the Workspace
- Export results as JSON (full metadata, base64 previews excluded) or CSV

Exported fields include:

| Field | Description |
|---|---|
| `filename` | Source image name |
| `model` | Canonical model ID (`midas_small`, `dpt_hybrid`, `dpt_large`) |
| `device` | Resolved runtime device string |
| `engine` | `pytorch` or `onnxruntime` |
| `latency_ms` | Server-side model timing in milliseconds |
| `abs_rel` | Absolute relative error against GT (requires GT upload) |
| `rmse` | Proxy RMSE from predicted mean, or GT RMSE when GT is provided |
| `delta_1` | δ < 1.25 accuracy threshold (requires GT) |
| `fallback` | Whether PyTorch fallback was used instead of ONNX |
| `warnings` | Any warnings from inference, GT alignment, or engine fallback |

---

### 7. 3D Reconstruction

<p align="center">
  <img src="docs/screenshots/3d-reconstruction.png" alt="3D Reconstruction" width="900">
</p>

The 3D tab converts a source image and its predicted depth into an approximate coloured point cloud using a pinhole camera projection model.

**Projection formula:**

```
X = (pixel_x − cx) × Z / focal_px
Y = (pixel_y − cy) × Z / focal_px   (negated for Y-up coordinate system)
Z = depth_normalised × depth_scale
```

Where `focal_px = focal_scale × max(width, height)` and `(cx, cy)` is the image centre.

Export formats: `PLY` (ASCII, with optional RGB vertex colours) and `OBJ` (Wavefront, with fractional RGB).

Available controls:

| Option | Purpose |
|---|---|
| Max dimension | Resize source image before depth inference |
| Max points | Total point budget for the exported file |
| Preview points | Separate budget for the in-browser WebGL preview (lighter than export) |
| Sampling | `grid` (deterministic stride) or `random` (fixed seed 0, reproducible) |
| Coordinate system | `y_up` (default, flips Y) or `camera` (raw projection) |
| Include RGB colors | Embeds source-image pixel colours per point vertex |
| Focal scale | Scales the assumed focal length; higher values flatten perspective |
| Depth scale | Multiplies Z values; scales the apparent scene depth |
| Near/far percentiles | Clips extreme depth outliers before normalisation |

> Monocular point clouds are approximate. The depth is relative, not metric. For accurate 3D reconstruction, camera calibration and a metric depth source are required.

---

### 8. Guide — Offline In-App Reference

<p align="center">
  <img src="docs/screenshots/guide-offline-reference.png" alt="Guide — Offline In-App Reference" width="900">
</p>

The Guide tab provides a fully offline accordion reference covering the complete workflow, metric definitions, model trade-offs, 3D reconstruction parameters, and troubleshooting steps. It does not call the backend and works even when the inference engine is offline.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## System Architecture

DepthLens Pro is split into a desktop shell, a local inference HTTP server, a model runtime, and a cache/storage layer.

The static diagram below gives a quick visual overview; the Mermaid diagram that follows shows the same system in more implementation detail.

<p align="center">
  <img src="docs/architecture/depthlens-system-architecture.svg" alt="DepthLens Pro local-first architecture diagram" width="900">
</p>

```mermaid
flowchart TB
    User["User"] --> Renderer["Electron Renderer\nHTML · CSS · JavaScript"]

    subgraph Electron["Electron Desktop App"]
        Main["Main Process\nWindow lifecycle · backend spawn · port checks"]
        Preload["Preload Bridge\nContext isolation · narrow IPC surface"]
        Renderer
    end

    subgraph API["FastAPI Backend"]
        Live["/live\nLightweight liveness"]
        Ready["/ready\nRuntime dependency readiness"]
        Health["/health\nFull diagnostics"]
        Routes["/estimate · /batch · /reconstruct · /benchmark"]
        Inference["Inference Service\nImage decode · model dispatch · depth encode"]
        Diagnostics["Diagnostics\nONNX · devices · cache · telemetry"]
    end

    subgraph Runtime["Model Runtime"]
        Registry["Model Registry\nCanonical IDs · aliases · ONNX paths"]
        Torch["PyTorch Torch Hub\nMiDaS / DPT — lazy-loaded, per-device lock"]
        ONNX["ONNX Runtime\nOptional acceleration — provider auto-selection"]
        GT["Ground Truth Evaluator\nResize · mask · median-scale align · metrics"]
        Reconstruct["Point Cloud Builder\nPinhole projection · PLY / OBJ"]
    end

    subgraph Cache["Local Cache"]
        Memory["In-process LRU dict\nThread-safe · max 256 entries"]
        Redis["Optional Redis\nTTL-based · connection pool · backoff on failure"]
        OnnxFiles["models/onnx/*.onnx\nLocally generated, not committed"]
    end

    Main -->|"spawns uvicorn (shell=false)"| API
    Main -->|"polls /live every ~1.2 s"| Live
    Renderer -->|"localhost fetch"| Routes
    Routes --> Inference
    Routes --> Diagnostics
    Inference --> Registry
    Registry --> Torch
    Registry --> ONNX
    ONNX --> OnnxFiles
    Inference --> GT
    Inference --> Reconstruct
    Inference --> Memory
    Inference --> Redis
```

A single `/estimate` request starts in the renderer when the user submits an image. The renderer sends a localhost request through the backend URL exposed by the preload bridge; Electron main owns backend startup and port discovery, while the FastAPI route validates upload size, model, colormap, metrics, output mode, and optional GT fields. The inference service decodes and possibly downsizes the image, builds a stable cache key from the full parameter tuple plus the image-content SHA-1, and checks Redis first when configured. Redis outages do not fail user requests; the service falls back to an in-memory LRU so inference remains available on local machines without external services. On a cache miss, dispatch enters the global `INFERENCE_MAX_CONCURRENCY` semaphore, then the selected PyTorch or ONNX model/device forward lock. The raw depth plane is resized, normalized, colorized/encoded, recorded in bounded telemetry, cached when eligible, and returned to the renderer for preview and export.

### Layer Responsibilities

| Layer | Key files | Responsibility |
|---|---|---|
| Electron main process | `electron-app/main.js`, `electron-app/src/main/*.js` | Small composition entrypoint plus focused modules for paths, backend HTTP probes, port fallback, PID metadata, Python resolution, backend lifecycle, settings persistence, and windows. Stable desktop lifecycle, backend control, and IPC wiring. |
| Renderer UI | `frontend/index.html`, `frontend/js/*.js`, `style.css` | Workspace tabs, charts, uploads, previews, 3D viewer, status orb, guide. Ordered frontend modules provide UI tabs, DOM integration, CSS-driven presentation, endpoint calls, persistence, and feature behavior. |
| Preload bridge | `electron-app/preload.js` | Narrow `contextBridge` surface — backend URL, dialogs, platform info only |
| Security policy | `electron-app/src/security-policy.js` | Navigation allowlist: local frontend file and `127.0.0.1:PORT` only |
| Process policy | `electron-app/src/backend-process-policy.js` | Checks command-line and stored PID metadata before terminating backend processes |
| FastAPI app | `backend/main.py`, `backend/api/` | Routes, CORS, JSON logging, exception handling, async lifespan hooks |
| Inference service | `backend/services/inference.py` | Image decoding, model dispatch (PyTorch + ONNX), depth normalisation, colourisation, encoding, depth-plane LRU cache |
| Model registry | `backend/model_registry.py` | Canonical IDs, display names, alias normalisation, ONNX path resolution across four candidate directories |
| Cache service | `backend/services/cache_service.py` | Redis integration with backoff, in-memory LRU fallback, versioned JSON serialisation (no pickle) |
| Diagnostics | `backend/services/diagnostics.py`, `onnx_diagnostics.py` | Module importability checks, provider discovery, ONNX weight validation |
| Reconstruction | `backend/services/reconstruction.py` | Pinhole point-cloud generation, PLY/OBJ serialisation, preview point downsampling |
| Ground truth | `backend/services/ground_truth.py` | GT decoding (PNG/TIFF/NPY), invalid-pixel masking, nearest-neighbour resize, median-scale alignment, benchmark metrics |

### Inference Concurrency Model

The backend uses two concurrency controls in combination:

1. **`INFERENCE_MAX_CONCURRENCY` semaphore** — an `asyncio.Semaphore` that limits the number of concurrent `/estimate` requests dispatched via `run_in_threadpool`. Default: 2.
2. **Per-model/device forward lock** — a `threading.Lock` stored in `_MODEL_FORWARD_LOCKS[f"{model_id}:{device_str}"]`. This prevents concurrent forward calls on the same model instance, which is unsafe on some torch backends. The lock covers only the `model(batch)` call; preprocessing and the bicubic resize run concurrently.

For ONNX, a matching `_ONNX_FORWARD_LOCKS` dict serialises `session.run()` calls per model/device pair.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## System Design Decisions

The system design favors transparent local execution over cloud dependency. These decisions keep the desktop app useful on typical developer and reviewer machines while still supporting optional acceleration and reproducible backend workflows.

### Caching

Cache entries are keyed by the model, colormap, device, metrics mode, requested outputs, maximum image dimension, and an image-content hash. Redis is used when configured because it provides TTL-based shared cache behavior across backend workers or repeated sessions. If Redis is not available, the in-memory LRU fallback keeps the app usable without any external service. Ground truth uploads bypass cache so benchmark and evaluation results are recomputed from the current GT input instead of reusing stale image-only outputs.

The tradeoff is that caching improves repeated runs and interactive iteration, but image-specific outputs and benchmark correctness require careful invalidation and bypass rules.

### Concurrency

FastAPI routes can receive multiple requests, but `INFERENCE_MAX_CONCURRENCY` limits how many inference jobs are dispatched at once. Model/device forward locks protect PyTorch and ONNX Runtime session calls for the same model/device pair, while preprocessing and non-critical work can still overlap where safe. This separates request-level concurrency from runtime safety for model execution.

The tradeoff is that maximum throughput is intentionally balanced against local machine stability, GPU/CPU memory pressure, and backend safety.

### Fallback and resilience

Redis is optional; Redis failures should not make local inference fail. Backend health endpoints separate lightweight liveness from deeper readiness and diagnostic checks, so the desktop shell can distinguish "process is reachable" from "all optional runtimes are ready." The Electron main process owns backend startup, port checks, and process lifecycle. Settings persistence uses safer local persistence behavior with recovery and fallback paths.

The tradeoff is that local-first desktop apps need graceful degradation because users may not have Redis, ONNX files, GPU drivers, or identical platform environments.

### Packaging and deployment

Native Electron builds are optimized for the desktop user experience. Docker Compose supports backend + Redis workflows for reproducible backend use without making Docker required for normal desktop inference. ONNX files are locally generated and validated rather than blindly committed, and platform/resource/ONNX resolvers keep packaging logic centralized.

The tradeoff is that packaged desktop convenience must be balanced with large ML assets, platform-specific native dependencies, and optional acceleration providers.

### Local-first privacy

Images are processed on localhost, and no cloud upload or API key is required for normal inference. Telemetry and observability should avoid raw images, image hashes, local paths, filenames, or high-cardinality user data so diagnostics stay useful without exposing private inputs.

The tradeoff is that privacy is prioritized over cloud scalability.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Quick Start

### Prerequisites

| Tool | Version | Why it is needed |
|---|---:|---|
| Git | Any recent | Clone the repository |
| Python | 3.10 – 3.12 | FastAPI backend and ML runtime |
| Node.js | LTS recommended | Electron desktop app |
| npm | Comes with Node.js | Install Electron dependencies |
| Docker | Optional | Backend + Redis containerised flow |

Check your tools:

```bash
git --version
python3 --version   # must be 3.10–3.12
node --version
npm --version
```

---

### Fastest Local Setup

```bash
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

# Install Python + Node dependencies, cache PyTorch MiDaS Torch Hub assets,
# and cache detector weights once. ONNX export is intentionally skipped here.
npm run setup

# Terminal 1 — start local FastAPI backend
npm run backend:dev

# Terminal 2 — open Electron desktop app
npm run frontend:dev
```

Verify the backend is live:

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
```

Expected `/live` response:

```json
{
  "status": "ok",
  "service": "DepthLens Pro API",
  "version": "3.1.0",
  "state": "idle",
  "pid": 12345,
  "uptime_seconds": 3.142
}
```

> **Setup-time model cache:** The setup step downloads and validates the PyTorch MiDaS Torch Hub repo, transforms, and checkpoints for MiDaS Small, DPT Hybrid, and DPT Large under `models/torch-cache`. It also caches RGB detector weights when enabled. ONNX files are separate and optional under `models/onnx`.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Installation Guide

| Path | Best for | What runs |
|---|---|---|
| A. Native desktop app | Normal desktop use on supported ARM64 systems | Packaged Electron app + embedded backend |
| B. Local development | Editing UI or backend code with hot-reload | Manual uvicorn + Electron dev shell |
| C. Backend only | API testing, scripting, CI, integrations | FastAPI server only (no Electron) |
| D. Docker Compose | Containerised backend with Redis cache | Backend container + Redis container |
| E. ONNX acceleration | Faster inference experiments | Locally generated `.onnx` weight files |

---

### A. Native Desktop App

Platform support is explicit and architecture-specific. The setup entry points are `scripts/setup-macos.sh`, `scripts/setup-linux.sh`, and `scripts/setup-windows.ps1` (also exposed through the Electron npm setup scripts).

| Platform / architecture | Status | Build command |
|---|---|---|
| macOS Apple Silicon arm64 | Supported | `npm run build:mac:arm64` |
| Intel Mac / macOS x64 | Not supported | `npm run build:mac:x64` exits with a clear error |
| macOS universal | Not supported | `npm run build:mac:universal` exits with a clear error |
| Windows arm64 | Supported | `npm run build:win:arm64` |
| Windows x64 | Supported | `npm run build:win:x64` |
| Linux arm64 | Supported | `npm run build:linux:arm64` |
| Linux x64 | Supported | `npm run build:linux:x64` |

Windows arm64 and x64 are supported, Linux arm64 and x64 are supported, and macOS remains Apple Silicon only.

The native workflow is deliberately split into **four repeatable steps per platform**: clone, setup, build, and launch. Setup is the only normal step that performs heavyweight dependency installs or model downloads. Standard setup installs the Python venv, backend dependencies, Electron dependencies, PyTorch MiDaS Torch Hub cache, and detector weights; it does **not** generate ONNX by default. ONNX setup adds export/validation for all three files in `models/onnx`: `midas_small.onnx`, `dpt_hybrid.onnx`, and `dpt_large.onnx`. Standard builds require `models/torch-cache` and treat ONNX as optional; ONNX builds require both the PyTorch cache and all three ONNX files.

#### macOS — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac # Install macOS dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:mac:arm64 # Build the macOS Apple Silicon native app
npm run launch:mac # Launch the packaged macOS app
```

</details>

#### macOS — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac:onnx # Install macOS dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:mac:arm64:onnx # Build the macOS Apple Silicon native app with ONNX resources required
npm run launch:mac # Launch the packaged macOS app
```

</details>

#### Windows ARM64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:win:arm64 # Build the Windows ARM64 native app
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Windows ARM64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:win:arm64:onnx # Build the Windows ARM64 native app with ONNX resources required
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Windows x86/x64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:win:x64 # Build the Windows x64/x86_64 native app
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Windows x86/x64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:win:x64:onnx # Build the Windows x64/x86_64 native app with ONNX resources required
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Linux ARM64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:linux:arm64 # Build the Linux ARM64 native app
npm run launch:linux # Launch the packaged Linux app
```

</details>

#### Linux ARM64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:linux:arm64:onnx # Build the Linux ARM64 native app with ONNX resources required
npm run launch:linux # Launch the packaged Linux app
```

</details>

#### Linux x86/x64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:linux:x64 # Build the Linux x64/x86_64 native app
npm run launch:linux # Launch the packaged Linux app
```

</details>

#### Linux x86/x64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:linux:x64:onnx # Build the Linux x64/x86_64 native app with ONNX resources required
npm run launch:linux # Launch the packaged Linux app
```

</details>


#### Setup report and diagnostics

After a successful setup, `scripts/doctor.py` writes a machine-readable report to `.depthlens/setup-report.json`. The report records the detected platform and CPU architecture, selected Python executable, virtualenv path, Node/npm versions, PyTorch MiDaS cache status, detector cache status, ONNX status, and the exact resource verification command that setup used. This file is diagnostic only: build scripts still verify the actual files in `models/torch-cache` and `models/onnx` instead of trusting the report.

Setup is safe to rerun. If pip, npm, MiDaS prefetch, detector prefetch, ONNX handling, or resource verification fails, rerun the platform setup command printed by the failure message. Use the standard setup command for PyTorch builds (`npm run setup:mac`, `npm run setup:linux`, or `npm run setup:win`) and the ONNX setup command when all three ONNX files are required (`npm run setup:mac:onnx`, `npm run setup:linux:onnx`, or `npm run setup:win:onnx`). Passing `--offline` validates existing caches only and does not download model assets; `--onnx-validate-only` validates existing ONNX files and never exports new ones.

#### Setup progress and verification

Setup output is intentionally verbose and streams in real time. Long-running installs and downloads print section headers and commands before they run, for example:

```text
[1/8] Selecting Python
[2/8] Creating or validating venv
[3/8] Upgrading pip/setuptools/wheel/certifi
[4/8] Installing backend dependencies
[5/8] Installing Electron dependencies
[6/8] Caching PyTorch MiDaS assets
[7/8] Handling optional ONNX assets
[8/8] Verifying resources
```

During MiDaS caching, `scripts/prefetch-midas-assets.py` prints the active `TORCH_HOME`, each selected model, whether offline validation is being used, retry counts, and final cache verification. Use these checks after setup:

```bash
npm run verify:resources
node electron-app/scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx optional .
npm run verify:onnx                 # validates existing ONNX files only
node electron-app/scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx require-all --models all .
```

Resource verification also supports structured JSON diagnostics without changing the default human-readable output. The JSON report includes the root kind, verification mode, ONNX mode, model readiness, detector readiness, Python candidate checks, and remediation text. The optional manifest schema in `models/resource-manifest.schema.json` documents the expected resource groups for diagnostics; the verification scripts still inspect the actual filesystem and do not trust the manifest as the sole source of truth.

```bash
node electron-app/scripts/verify-resources.js --json --root-kind repo --mode native --torch-cache required --onnx optional .
npm run verify:resources
npm run verify:onnx
```

Packaged resource verification is performed by the build scripts. To run it manually after packaging, use:

```bash
cd electron-app
node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --torch-cache required --onnx optional
node scripts/verify-packaged-resources.js --platform linux --arch x64 --mode native --torch-cache required --onnx require-all --models all
```

Model assets are intentionally not committed to git. They live in:

| Asset | Repo path | Packaged path | Required for standard builds |
|---|---|---|---|
| PyTorch MiDaS Torch Hub repo, transforms, checkpoints | `models/torch-cache` | `<Resources>/models/torch-cache` | Yes |
| Detector checkpoints | `models/torch-cache/hub/checkpoints` | `<Resources>/models/torch-cache/hub/checkpoints` | Optional feature cache |
| ONNX exports | `models/onnx` | `<Resources>/models/onnx` | No; required only for ONNX builds/benchmarks |

Platform-specific outputs:

| Platform | Output |
|---|---|
| macOS | `electron-app/dist/mac-arm64/DepthLens Pro.app` and `.dmg` |
| Windows | `electron-app/dist/win-arm64-unpacked/` or `electron-app/dist/win-x64-unpacked/` and NSIS installer |
| Linux | `electron-app/dist/*arm64*.AppImage`, `electron-app/dist/*x64*.AppImage`, and unpacked resources when retained by electron-builder |


#### Packaging size preflight and automatic DMG sizing

Every native packaging command now runs `electron-app/scripts/package-size-preflight.js` before Electron Builder starts. The preflight is deterministic and cross-platform: it walks the Electron app, backend, frontend, Python `venv`, `models/torch-cache`, `models/onnx`, and the local Electron runtime cache while ignoring avoidable junk such as `__pycache__`, `.pyc`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.git`, logs, temporary files, and source maps. It prints a per-directory footprint summary, checks free space for the host temp/build/output locations, and fails before packaging if required resources are missing or there is not enough room for a large offline artifact.

For macOS DMG builds, the same preflight computes `dmg.size` automatically instead of relying on a hand-edited hardcoded value. The planned DMG size is the estimated payload size plus a configurable safety margin and filesystem overhead, rounded up to MiB with a sane minimum. The wrapper passes that value to Electron Builder for `npm run build:mac:arm64` and `npm run build:mac:arm64:onnx`, so large bundles containing Electron, the Python virtualenv, Torch/TorchVision, ONNX Runtime, MiDaS Torch cache, and all ONNX exports get a DMG volume sized for the actual payload.

Useful maintainer commands and knobs:

```bash
cd electron-app
npm run preflight:package -- --platform darwin --arch arm64 --onnx require-all --models all
node scripts/package-size-preflight.js --json --platform linux --arch x64 --onnx optional
```

| Environment variable | Default | Purpose |
|---|---:|---|
| `DEPTHLENS_PACKAGE_SIZE_MARGIN` | `0.35` | Fractional safety margin added to the estimated payload before macOS DMG sizing. |
| `DEPTHLENS_DMG_MIN_BYTES` | `2147483648` | Minimum computed macOS DMG size in bytes. |
| `DEPTHLENS_PACKAGE_MIN_FREE_BYTES` | `2147483648` | Minimum host free-space floor for temp/build/output checks; platform payload requirements can raise this automatically. |

Troubleshooting packaging size failures:

- If macOS DMG creation logs `ditto` errors under `/Volumes/DepthLens Pro ...` with `No space left on device`, that often means the temporary mounted DMG image is too small, not that the Mac startup disk is full. Re-run the preflight, inspect the computed `dmg.size`, and either increase `DEPTHLENS_PACKAGE_SIZE_MARGIN`/`DEPTHLENS_DMG_MIN_BYTES` or reduce bundled resources.
- If the preflight reports insufficient host temp/build/output disk space, free space in the listed directory or point the OS temp directory/build output at a larger volume before rerunning the same build command.
- If the preflight reports missing `models/onnx` files for an ONNX build, run `npm run setup:<platform>:onnx` and `npm run verify:onnx:required`; standard non-ONNX builds still require `models/torch-cache` but do not require ONNX exports.
- Large offline builds are expected to be several GiB because they include Electron, the Python runtime, PyTorch/TorchVision, ONNX Runtime, MiDaS Torch cache, and optional ONNX weights such as `dpt_large.onnx`. Avoid deleting required offline model assets; the package filters only exclude caches, logs, compiled Python bytecode, source maps, and temporary/setup diagnostics.

#### No silent downloads during build

The build scripts verify repo resources before packaging and packaged resources after packaging. They do not silently download model assets. If `models/torch-cache` is missing, standard builds fail early with a “run setup first” remediation. If an ONNX build is requested and any ONNX file is missing or empty, the build fails early with the matching `setup:<platform>:onnx` command. Electron packages `models` as extra resources, so packaged resources contain `Resources/models/torch-cache` and, for ONNX builds, `Resources/models/onnx`.

---

### B. Local Development

Run the backend and desktop shell in two separate terminals:

```bash
# Terminal 1
npm run backend:dev
# Equivalent: venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --reload

# Terminal 2
npm run frontend:dev
# Equivalent: cd electron-app && electron .
```

Useful inspection commands:

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/health      # full diagnostics including device list, ONNX status, memory/disk telemetry
curl http://127.0.0.1:8765/devices     # available compute targets
curl http://127.0.0.1:8765/onnx/status # ONNX weight and provider diagnostics
```

---

### C. Backend Only

```bash
npm run setup
npm run backend:dev
```

Or invoke Uvicorn directly (useful for custom ports or workers):

```bash
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --workers 1
```

The `backend.app` entry point (`backend/app.py`) is a backward-compatible ASGI shim that ensures the repo root is on `sys.path` before importing `backend.main:app`. Both of these are equivalent and interchangeable:

```bash
uvicorn backend.app:app    # repo-root CWD
uvicorn app:app            # backend/ CWD packaged compatibility flow
```

---

### D. Docker Compose

```bash
docker compose up --build          # foreground
docker compose up --build -d       # background
docker compose down                # stop
docker compose down -v             # stop and remove Redis volume
```

Verify:

```bash
curl http://127.0.0.1:8765/live
```

Docker defaults:

| Setting | Default |
|---|---:|
| Backend port | `8765` |
| Redis port | `6379` (internal to Compose network, not exposed) |
| CPU limit | `4.0` cores |
| Memory limit | `8 GB` |
| Shared memory | `8 GB` (needed for PyTorch multiprocessing) |
| Backend user | Non-root `depthlens` (UID/GID created in image) |

The Dockerfile uses a two-stage build: a `builder` stage installs all Python wheels into a venv at `/opt/venv`, and a `runner` stage copies only the venv and the backend package — keeping the final image free of build tools.

---

### E. Optional ONNX Acceleration

ONNX Runtime is entirely optional for standard builds and separate from the required PyTorch MiDaS cache. The app works without ONNX files by using PyTorch MiDaS assets cached under `models/torch-cache`; ONNX builds add `.onnx` files under `models/onnx`.

Generate ONNX files locally:

```bash
# Export the default MiDaS Small graph
venv/bin/python backend/scripts/export_onnx.py --model midas_small --force

# Export all supported models
venv/bin/python backend/scripts/export_onnx.py --all --force

# Validate existing ONNX files without re-exporting
npm run verify:onnx
```

The export script tries two strategies in order:

1. **Legacy `torch.onnx.export`** — standard ONNX opset-17 export with constant-folding.
2. **Dynamo export** — uses `torch.onnx.export(..., dynamo=True)` when the PyTorch version supports it.

If the first strategy produces an invalid graph (checked via `onnx.checker.check_model` and a dummy inference session), the file is quarantined with a `.failed` suffix and the second strategy is tried. This ensures partially-exported files never silently corrupt future benchmark runs.

Expected ONNX location (configurable via `ONNX_WEIGHTS_DIR` or `DEPTHLENSPRO_MODEL_DIR`):

```
models/onnx/
├── midas_small.onnx
├── dpt_hybrid.onnx
└── dpt_large.onnx
```

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Terminal-Only Dev Verification

This flow uses the project from the terminal in dev mode — it sets up dependencies, starts the FastAPI backend, checks the backend, then opens the Electron dev shell without creating a native installer/package. The repo exposes `backend:dev` and `frontend:dev` for this.

### macOS — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac # Install macOS dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### macOS — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac:onnx # Install macOS dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Windows ARM64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Windows ARM64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Windows x86/x64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows x64/x86_64 dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Windows x86/x64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows x64/x86_64 dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
$backend = Start-Process -FilePath "npm.cmd" -ArgumentList "run backend:dev" -PassThru # Start the FastAPI backend from terminal
Start-Sleep -Seconds 5 # Give the backend time to start
Invoke-RestMethod http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
Stop-Process -Id $backend.Id -Force # Stop the backend after closing the dev app
```

</details>

### Linux ARM64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux ARM64 dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Linux ARM64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux ARM64 dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Linux x86/x64 — Standard terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux x64/x86_64 dependencies and standard PyTorch model cache
npm run verify:resources # Verify standard resources
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/live # Check that the backend is live
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

### Linux x86/x64 — ONNX terminal-only test

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux x64/x86_64 dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run backend:dev & # Start the FastAPI backend from terminal
BACKEND_PID=$! # Store the backend process ID
sleep 5 # Give the backend time to start
curl http://127.0.0.1:8765/onnx/status # Check ONNX model/provider status
npm run frontend:dev # Open the dev app without building a native package
kill "$BACKEND_PID" 2>/dev/null || true # Stop the backend after closing the dev app
```

</details>

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Configuration

Settings are read from environment variables, with optional fallback to a `.env` file in the repository root. `pydantic-settings` is used when available; a lightweight dotenv parser handles the case where dependencies are not yet installed.

### Safe Local `.env`

```env
HOST=127.0.0.1
PORT=8765
LOG_LEVEL=INFO
DEBUG=false

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL_SECONDS=3600
CACHE_MAX_ENTRIES=256

DEPTHLENS_PRELOAD_MODEL=false
DEPTHLENS_WARMUP_MODEL=MiDaS_small
DEPTHLENS_WARMUP_DEVICE=auto
DEPTHLENS_MAX_DIM=1536
DEPTHLENS_DEFAULT_METRICS=fast
DEPTHLENS_DEFAULT_OUTPUTS=color
DEPTHLENS_OBSERVABILITY_ENABLED=true
DEPTHLENS_PROMETHEUS_ENABLED=true
DEPTHLENS_TELEMETRY_MAX_EVENTS=200
DEPTHLENS_TRACE_HISTORY_LIMIT=200
DEPTHLENS_CRASH_HISTORY_LIMIT=100
DEPTHLENS_BENCHMARK_HISTORY_LIMIT=50
DEPTHLENS_TRACE_SAMPLE_RATE=1.0
```

### Server

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` locally, `0.0.0.0` in Docker | ASGI bind host |
| `PORT` | `8765` | ASGI port |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `DEBUG` | `false` | Enables FastAPI debug mode (detailed error responses) |
| `WEB_CONCURRENCY` | `1` | Uvicorn worker count (Docker only; use 1 for single-GPU inference) |

### Cache

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | unset | Full Redis URL override (takes precedence over individual fields) |
| `REDIS_HOST` | `127.0.0.1` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis logical database |
| `REDIS_PASSWORD` | unset | Optional Redis password |
| `REDIS_SOCKET_TIMEOUT_SECONDS` | `1.5` | Connect/read timeout; keep low to fail fast and fall back to in-memory |
| `REDIS_MAX_CONNECTIONS` | `20` | Connection pool maximum |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `CACHE_MAX_ENTRIES` | `256` | In-memory LRU entry cap |

### Inference

| Variable | Default | Description |
|---|---|---|
| `DEPTHLENS_PRELOAD_MODEL` | `false` | Warm a model in the background after startup |
| `DEPTHLENS_WARMUP_MODEL` | `MiDaS_small` | Model to pre-warm when preload is enabled |
| `DEPTHLENS_WARMUP_DEVICE` | `auto` | Device to pre-warm on |
| `DEPTHLENS_SKIP_WARMUP` | unset | Set to `1` to skip warmup (used in CI/testing) |
| `DEPTHLENS_MAX_DIM` | `1536` | Maximum long image edge before down-scaling |
| `DEPTHLENS_DEFAULT_METRICS` | `fast` | `none`, `fast`, or `full` |
| `DEPTHLENS_DEFAULT_OUTPUTS` | `color` | `color`, `gray`, or `color,gray` |
| `INFERENCE_MAX_CONCURRENCY` | `2` | Max concurrent inference operations (asyncio semaphore) |
| `DEPTHLENS_OBSERVABILITY_ENABLED` | `true` | Enable local observability snapshots and instrumentation |
| `DEPTHLENS_PROMETHEUS_ENABLED` | `true` | Enable `/metrics` Prometheus exposition when `prometheus-client` is available |
| `DEPTHLENS_TELEMETRY_MAX_EVENTS` | `200` | Bounded recent HTTP/inference event history size |
| `DEPTHLENS_TRACE_HISTORY_LIMIT` | `200` | Bounded trace span history size |
| `DEPTHLENS_CRASH_HISTORY_LIMIT` | `100` | Bounded sanitized crash history size |
| `DEPTHLENS_BENCHMARK_HISTORY_LIMIT` | `50` | Bounded benchmark history size |
| `DEPTHLENS_TRACE_SAMPLE_RATE` | `1.0` | Trace sampling ratio from `0.0` to `1.0` |
| `ORT_INTRA_OP_NUM_THREADS` | CPU-dependent / Docker `2` | ONNX Runtime intra-op thread pool |
| `ORT_INTER_OP_NUM_THREADS` | `1` | ONNX Runtime inter-op thread pool |

### Paths

| Variable | Default | Description |
|---|---|---|
| `DEPTHLENS_BACKEND_PORT` | `8765` | Electron backend port hint (read before spawning uvicorn) |
| `DEPTHLENSPRO_MODEL_DIR` | unset | Custom model directory; ONNX files searched in `{dir}/onnx/` |
| `DEPTHLENS_ONNX_DIR` | unset | Direct ONNX directory override |
| `ONNX_WEIGHTS_DIR` | unset | Legacy ONNX directory (lowest priority) |
| `DEPTHLENS_AUTO_EXPORT_ONNX` | `false` | Auto-export ONNX during benchmark when weights are missing |

### Observability Privacy

Telemetry is local-only: DepthLens Pro does not send analytics to cloud services or external telemetry endpoints. Histories are bounded in process memory, Prometheus labels intentionally avoid high-cardinality user data, and telemetry avoids raw images, base64 payloads, uploaded filenames, image hashes, cache keys, local full paths, and private exception details.

### CI / Test Flags

| Variable | Purpose |
|---|---|
| `TESTING=1` | Lightweight test mode; skips warmup and model downloads |
| `CI=1` | CI marker used by test fixtures |
| `CODEX_ENV=1` | Automation/sandboxed environment marker |
| `DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1` | Prevents torch.hub from downloading weights (used in offline CI) |

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## API Reference

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

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Models, Colormaps & Metrics

### Supported Models

| Canonical ID | Display name | Architecture | Input size | Recommended use |
|---|---|---|---:|---|
| `midas_small` | MiDaS Small | MiDaS small / EfficientNet-Lite | 256×256 | Fast previews, CPU-only, webcam |
| `dpt_hybrid` | DPT Hybrid | DPT Hybrid / ViT-Hybrid | 384×384 | Balanced quality and speed |
| `dpt_large` | DPT Large | DPT Large / ViT-Large | 384×384 | Maximum detail; GPU required for practical speed |

Model names are normalised to canonical IDs automatically. All of these resolve to `midas_small`:

```
MiDaS_small  /  MiDaS Small  /  midas_small  /  midas-small  /  MiDaS-Small
```

---

### Which Model Should I Use?

| Goal | Recommended model |
|---|---|
| Fastest result | MiDaS Small |
| Webcam / real-time preview | MiDaS Small |
| Balanced visual quality | DPT Hybrid |
| Maximum edge detail | DPT Large |
| CPU-only machine | MiDaS Small |
| CUDA or MPS GPU available | DPT Hybrid or DPT Large |
| Benchmarking ONNX acceleration | MiDaS Small (most ONNX-export-friendly) |

---

### Colormaps

Supported colormaps:

```
inferno · plasma · viridis · magma · jet · hot · bone · turbo
```

| Colormap | Use when |
|---|---|
| `inferno` | You want high contrast; safe default for most visualisations |
| `viridis` | You want perceptually uniform, colorblind-safer output |
| `plasma` | You want a bright, warm presentation style |
| `magma` | You want a softer dark-to-light depth map |
| `turbo` | You want strong colour separation across the full depth range |
| `jet` | You need a classic rainbow map for compatibility |
| `hot` | You want heat-map-style output |
| `bone` | You want a subtle greyscale-adjacent map |

---

### Metrics Modes

| Mode | Description | Latency impact |
|---|---|---|
| `none` | Return output images only — no metric computation | Minimal |
| `fast` | Lightweight prediction statistics (min, max, mean, std, entropy, histogram) | ~1–3 ms |
| `full` | Full statistics plus proxy diagnostics (SSIM, SILog, PSNR, gradient, edge density) | ~5–15 ms |

### Metric Groups

| Group | Examples | Requires GT? |
|---|---|---|
| Prediction stats | min, max, mean, std, median, histogram, entropy, coverage | No |
| Proxy metrics | SSIM (vs input), SILog, PSNR, gradient error, edge density, MAE/RMSE vs predicted mean | No |
| Ground-truth metrics | Abs Rel, Sq Rel, GT RMSE, log RMSE, δ < 1.25 / 1.25² / 1.25³ | Yes |
| Reported unavailable | GT SSIM, GT PSNR, ordinal error, surface normal error, LPIPS | Depends; not yet implemented |

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Ground Truth Evaluation

DepthLens Pro supports GT-based evaluation for one image at a time.

### Supported GT Formats

| Format | Notes |
|---|---|
| PNG | Single-channel preferred; multi-channel converted to greyscale |
| TIFF / TIF | Single-channel float or integer depth |
| NPY | Numeric depth array loaded with `allow_pickle=False` |
| EXR / PFM | Not currently supported |

### GT Processing Flow

```mermaid
flowchart LR
    A["Upload image + GT file"] --> B["Decode GT as float32"]
    B --> C["Apply optional gt_scale multiplier"]
    C --> D["Mask invalid / non-finite / gt_invalid_value pixels"]
    D --> E["Nearest-neighbour resize GT to prediction shape"]
    E --> F["Median-scale align prediction to GT"]
    F --> G["Compute benchmark metrics over valid pixels"]
```

### Why Median-Scale Alignment?

MiDaS-style monocular depth is inherently **relative** — the model predicts depth ordering and relative scale, not absolute distances in metres. Directly comparing raw predictions against metric GT would produce meaningless numbers.

Median-scale alignment computes:

```
scale = median(gt_valid_pixels) / median(pred_positive_pixels)
pred_scaled = pred * scale
```

This removes the global scale ambiguity before computing error metrics, making results comparable across different scenes and model configurations. The scale factor is clamped to `[1e-3, 1000]` to reject implausible alignments (e.g. unit mismatches where GT is in millimetres but predictions are in normalised units).

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Understanding Depth Metrics

### Prediction-Only Metrics (no GT required)

These metrics are computed from the normalised depth plane alone. They measure the internal richness and structure of the prediction, not its accuracy against a reference.

| Metric | What it measures | Good values |
|---|---|---|
| **Entropy** | Shannon entropy of the 256-bin depth histogram | Higher = more uniformly distributed depth values |
| **Dynamic range** | log₂(max/min non-zero depth) in bits | Higher = wider depth variation captured |
| **Coverage** | Fraction of histogram bins with ≥1% of peak count | Higher = depth values spread across full range |
| **Edge density** | Fraction of pixels with gradient magnitude > mean+std | Higher = more structural depth edges |
| **SSIM (proxy)** | Structural similarity between predicted depth and greyscale RGB input | Not a benchmark metric; correlates depth structure with image edges |
| **SILog (proxy)** | Log-depth dispersion of the prediction itself | Not true SILog; use for relative comparison only |

### GT Metrics (requires ground truth upload)

These are standard monocular depth estimation benchmark metrics used in papers like Eigen et al. and the MiDaS evaluation suite.

| Metric | Formula | Interpretation |
|---|---|---|
| **Abs Rel** | `mean(|pred − gt| / gt)` | Primary quality metric; lower is better |
| **Sq Rel** | `mean((pred − gt)² / gt)` | Penalises large errors more heavily; lower is better |
| **GT RMSE** | `sqrt(mean((pred − gt)²))` | Root mean squared error; lower is better |
| **GT Log RMSE** | `sqrt(mean((log pred − log gt)²))` | Less sensitive to scale outliers; lower is better |
| **δ < 1.25** | `mean(max(pred/gt, gt/pred) < 1.25)` | Percentage of pixels within 25% of GT; higher is better |
| **δ < 1.25²** | Same with threshold 1.5625 | Looser accuracy; higher is better |
| **δ < 1.25³** | Same with threshold 1.953 | Loosest threshold; higher is better |

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Testing & CI

### Run Local Checks

Install backend dependencies before full pytest collection. Use the normal four-step workflow (`clone → setup → build
→ launch`) and run the appropriate setup command first, for example
`npm run setup:<platform>` for standard builds or `npm run setup:<platform>:onnx`
when validating the required ONNX files (`midas_small.onnx`, `dpt_hybrid.onnx`,
and `dpt_large.onnx`). Standard setup/build does not require ONNX generation.

```bash
scripts/ci.sh workflow-policy
scripts/ci.sh docs-contract
scripts/ci.sh backend-quality
scripts/ci.sh electron-contract
scripts/ci.sh docker-build
scripts/ci.sh all
```

The `scripts/ci.sh` entrypoint exports CI-safe defaults that disable model downloads and warmup, use the in-memory cache backend, and keep checks non-interactive. `docker-build` is the only subcommand that requires Docker; it fails fast with an actionable message when Docker is unavailable in a local or Codex sandbox.

### Useful Test Commands

```bash
# Backend tests only
pytest backend/tests/

# One test file with verbose output
pytest backend/tests/test_routes.py -v

# Electron lightweight security and resource tests
cd electron-app && npm test
```

### CI Pipeline

GitHub Actions uses a dynamic workflow named `CI`. It runs for pull requests targeting `main`, pushes to `main`, and manual `workflow_dispatch` runs. It does not run push CI for every feature or Codex branch. A fast `detect-changes` job classifies changed files first, then only relevant jobs run:

- docs-only changes run `docs-contract` and the final gate, without expensive backend, Electron, or Docker work.
- backend or Python tooling changes run `backend-quality`.
- Electron/frontend or Node tooling changes run `electron-contract`.
- Dockerfile, backend runtime, requirements, compose, or Docker ignore changes run the cached Buildx `docker-build`.
- workflow/tooling changes, pushes to `main`, and manual `workflow_dispatch` runs are conservative and require full CI.

Branch protection should require only the stable final check named `ci-passed`. Internal jobs are intentionally dynamic and may be skipped when `detect-changes` proves they are irrelevant. The `ci-passed` job validates that required jobs succeeded and emits clear errors if a required job failed, was cancelled, timed out, or was skipped. Docker might not be available in Codex or a local sandbox, but GitHub Actions runs Docker with `docker/setup-buildx-action` and `docker/build-push-action` when Docker-related changes require it.

The test suite covers API behaviour, cache serialisation safety (no pickle deserialization), ONNX fallback paths, reconstruction logic, packaging verification, and Electron security policies — without requiring a GPU, Redis instance, or real model weights.

#### What the Tests Stub

- **torch / cv2** — fully stubbed via `conftest.py` and `monkeypatch`; no GPU or system OpenCV library required
- **ONNX Runtime** — stubbed per-test with `sys.modules` injection
- **Redis** — disabled via `monkeypatch.setattr(cache_service, "redis", None)`
- **Model downloads** — prevented via `DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1`
- **Warmup** — skipped via `TESTING=1`

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Production & Packaging

### Native Platform Builds

Use the root `npm run build:*` commands shown in the Installation Guide, or call the verbose platform scripts directly:

```bash
# macOS Apple Silicon only
scripts/build-native-macos.sh --arch arm64 --without-onnx

# Linux x64 / arm64
scripts/build-native-linux.sh --arch x64 --without-onnx
scripts/build-native-linux.sh --arch arm64 --without-onnx
```

```powershell
# Windows x64 / arm64
.\scripts\build-native-windows.ps1 -Arch x64 -WithoutOnnx
.\scripts\build-native-windows.ps1 -Arch arm64 -WithoutOnnx
```

ONNX variants download or generate all required ONNX files under `models/onnx` before packaging:

```bash
scripts/build-native-macos.sh --arch arm64 --with-onnx --onnx-models all
scripts/build-native-linux.sh --arch x64 --with-onnx --onnx-models all
```

```powershell
.\scripts\build-native-windows.ps1 -Arch x64 -WithOnnx -OnnxModels all
```

Each native build script:

1. Runs `setup-{platform}.sh` / `setup-windows.ps1` (creates venv, installs deps, checks Node)
2. Cleans previous `dist/` output
3. Runs `verify-resources.js` to confirm all required files are present before packaging
4. Invokes `electron-builder` for the target platform and architecture
5. Runs `verify-packaged-resources.js` to confirm the packaged app contains backend, frontend, venv, and models directories

Pass `--with-onnx` and `--onnx-models midas_small` to export and bundle ONNX weights in the package.

### Docker Backend

Build only:

```bash
docker build -t depthlenspro-backend:latest .
```

Run backend + Redis:

```bash
docker compose up --build        # foreground
docker compose up --build -d     # background
```

The Docker image uses Python 3.12 slim, installs dependencies into an isolated venv, and runs as a non-root `depthlens` user. The two-stage build keeps the final image free of build-time dependencies.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Troubleshooting

### Backend Offline

Check liveness:

```bash
curl http://127.0.0.1:8765/live
```

Run full diagnostics:

```bash
python scripts/diagnose_backend.py
```

Common fixes:

```bash
npm run backend:dev     # start the backend
npm run stop:backend    # terminate a stale backend process
```

---

### `/live` Works but `/ready` Fails

`/live` only confirms the HTTP process is responding. `/ready` checks whether all required Python modules (`fastapi`, `uvicorn`, `numpy`, `torch`, `cv2`, `PIL`) can be imported without error.

```bash
npm run setup                       # reinstall dependencies
venv/bin/python -m pip check        # verify no broken packages
curl http://127.0.0.1:8765/ready    # check readiness again
```

---

### Inference Controls Are Disabled

Usually this means the UI is waiting for the backend's `/ready` endpoint to confirm inference runtime availability.

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/health
```

Then restart:

```bash
npm run stop:backend
npm run backend:dev
```

---

### Renderer Startup Initialization

Renderer startup is dependency-ordered and failure-isolated in both packaged Electron installs and development runs. Foundational state helpers now load before chart/decorative scripts, and optional renderer initialization such as background canvas animation, Chart.js setup, Compare controls, pointer glow, guide accordion, and scroll navigation is isolated from backend startup. If a decorative or chart dependency is unavailable, the app should log a console warning while backend URL resolution, `/live`, `/ready`, `/health` diagnostics, device discovery, polling, and the Depth Engine status panel continue to initialize normally.

---

### “Depth engine ready” but inference fails

`/ready` separates `backend_alive`, `runtime_imports_ready`, `model_assets_ready`, `pytorch_hub_cache_ready`, `onnx_any_ready`, `onnx_all_ready`, and `inference_ready`. If `inference_ready` is false, inspect `fatal_reason` and `recommended_action`:

```bash
curl http://127.0.0.1:8765/ready
```

For native apps, rerun the platform setup and rebuild so packaged resources include the cache:

```bash
npm run setup:mac && npm run build:mac:arm64
npm run setup:linux && npm run build:linux:x64
npm run setup:win && npm run build:win:x64
```

### `MODEL_ASSETS_UNAVAILABLE`

This means MiDaS Torch Hub repo code or checkpoints are missing/incomplete under `TORCH_HOME` (`models/torch-cache` in the repo, `<Resources>/models/torch-cache` in packaged apps). The response includes `torch_home`, `expected_cache`, and an action. ONNX is optional for standard builds; PyTorch MiDaS assets are not.

Fix:

```bash
npm run setup:mac        # or setup:linux / setup:win
npm run build:mac:arm64 # or your platform build command
```

### `ONNX missing_model_file`

This is not a core engine failure for standard builds. It means the optional ONNX benchmark/acceleration file is absent. Standard inference still uses PyTorch when `models/torch-cache` is valid. For ONNX builds or benchmarks, generate all required files:

```bash
npm run setup:mac:onnx      # or setup:linux:onnx / setup:win:onnx
npm run verify:onnx
```

### Setup appears stuck

Setup should no longer sit silently at “loading assets.” It streams pip, npm, MiDaS, detector, ONNX export, and verification output in real time. If progress stops, the last printed command/model identifies the current operation. Re-run with a larger MiDaS timeout if a slow network is expected:

```bash
python scripts/doctor.py --timeout-seconds 1800 --retries 3
```

Rerunning setup is resumable and safe: it reuses the existing `venv`, `electron-app/node_modules`, `models/torch-cache`, and `models/onnx` when they validate. If you need a no-network check, use offline validation without downloads:

```bash
venv/bin/python scripts/prefetch-midas-assets.py --offline --models all
```

### Packaged app missing resources

Build scripts verify packaged resources, but installed stale copies can still be launched accidentally. Verify the packaged output directly:

```bash
cd electron-app
node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --torch-cache required --onnx optional
node scripts/verify-packaged-resources.js --platform win32 --arch x64 --mode native --torch-cache required --onnx optional
node scripts/verify-packaged-resources.js --platform linux --arch x64 --mode native --torch-cache required --onnx optional
```

### RGB Camera Detection Reports Missing Detector Weights

The RGB Camera View / 3D tab uses TorchVision detector weights cached by the setup step under `models/torch-cache`. This cache is not an ONNX cache; ONNX remains optional under `models/onnx`.

If `/api/detect` or RGB Camera View reports missing detector weights, rerun the setup step with network access for your platform:

```bash
npm run setup:mac
npm run setup:linux
npm run setup:win
```

Only skip detector weights if RGB object detection is not needed:

```bash
python scripts/doctor.py --without-detector-weights
```

When skipped, setup completes but RGB Camera detection may fail until detector weights are cached.

---

### ONNX Benchmark Is Unavailable

Validate existing ONNX files:

```bash
npm run verify:onnx
```

Check backend ONNX diagnostics:

```bash
curl http://127.0.0.1:8765/onnx/status
```

The response includes `recommended_export_command` with the exact CLI invocation needed to generate any missing file.

Generate ONNX weights:

```bash
venv/bin/python backend/scripts/export_onnx.py --model midas_small --force
```

The app continues working through PyTorch fallback. ONNX is never required for the core depth estimation workflow.

---

### Port `8765` Is Already in Use

Diagnose:

```bash
python scripts/diagnose_backend.py
```

macOS / Linux:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

Windows PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen
```

Use a different port:

```bash
DEPTHLENS_BACKEND_PORT=8770 npm run frontend:dev
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8770
```

Electron automatically finds the next available port if `DEPTHLENS_BACKEND_PORT` is not pinned and the default port is busy. The frontend reads the actual backend URL from Electron's IPC at runtime, so port mismatches between the shell and the app are avoided.

---

### Packaged App Missing Resources

Verify resources before packaging:

```bash
npm run verify:resources
```

Verify a packaged output after building:

```bash
cd electron-app
npm run verify:packaged:mac     # darwin arm64
npm run verify:packaged:win     # win32 arm64
npm run verify:packaged:linux   # linux arm64
```

---

### macOS Duplicate App Instances / Spotlight Conflicts

```bash
cd electron-app
npm run scan:apps       # lists all DepthLens Pro.app bundles found on disk
npm run clean:dist      # removes electron-app/dist/
npm run clean:install   # removes /Applications/DepthLens Pro.app
```

Only ever launch `electron-app/dist/mac-arm64/DepthLens Pro.app`. A bundle at `dist/mac/` (without the `arm64` suffix) is a stale build artefact that should be removed.

---

### PowerShell Blocks Scripts

Use the npm wrapper (no execution-policy change required):

```powershell
npm run setup:win
```

Or run with an explicit policy bypass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1
```

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Security

DepthLens Pro is designed as a local-first desktop ML tool. The security model assumes the inference server runs on `127.0.0.1` and is accessed only by the local Electron renderer — not exposed to the network.

### Security Design

| Area | Approach |
|---|---|
| Local inference | All requests go to `127.0.0.1`; no hosted inference service is used |
| Renderer isolation | Electron `contextIsolation: true`, `sandbox: true`, `nodeIntegration: false` |
| Navigation policy | Renderer navigation is restricted to the local frontend file and `127.0.0.1:PORT` only — other localhost ports are blocked |
| External links | HTTPS and `mailto:` links open in the system browser via `shell.openExternal`; new-window requests are denied |
| Backend process ownership | Before killing any process on the backend port, Electron checks that the process command-line and stored PID metadata match a known DepthLens-owned invocation |
| Single instance | Electron prevents multiple desktop app instances from fighting over backend state |
| PID metadata | Backend PID and connection metadata are stored in platform user-data files at mode `0600` |
| Cache serialisation | Cache payloads are serialised as versioned JSON (magic prefix `DLP2\0`). Legacy pickle payloads (prefix `DLP1\0` or `\x80`) are detected, deleted, and never deserialised |
| Error handling | Client-facing 500 responses are sanitised (`"Internal server error"` only); full stack traces remain in server logs |
| Secrets | Default local flow requires no API keys, tokens, or credentials |
| Spawn safety | Backend is started with `spawn(pythonPath, args, { shell: false })` — arguments are passed as an array, not interpolated into a shell string |

### Privacy Notes

- Uploaded images are processed locally and never leave the machine.
- The backend listens on `127.0.0.1` by default; Docker mode exposes the port according to your Compose port mapping.
- First-time PyTorch model loading may download model weights from Torch Hub (GitHub/CDN) if they are not already cached at `~/.cache/torch/hub`.
- ONNX files are generated locally and stored under `models/onnx/`.

### Reporting Vulnerabilities

Please do **not** open a public GitHub issue for security-sensitive reports.

Include:

- Description of the issue
- Steps to reproduce
- Affected component
- Possible impact
- Suggested mitigation, if known

See [`SECURITY.md`](SECURITY.md) for the full policy.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Project Structure

<details>
<summary>Show full repository tree</summary>

```text
DepthLensPro/
├── backend/
│   ├── api/
│   │   ├── live.py                  # / and /live routes (no heavy imports)
│   │   ├── routes.py                # Thin route orchestration — estimate, batch, health, benchmark, cache, reconstruct
│   │   ├── validation.py            # Request field, upload, model, colormap, metrics/output validation helpers
│   │   ├── errors.py                # Public HTTP error payload mapping helpers
│   │   ├── device_state.py          # Cached device discovery, accelerator checks, readiness diagnostics
│   │   └── system_telemetry.py      # Memory/disk telemetry helpers used by /health
│   ├── core/
│   │   └── paths.py                 # Central repo/model/Torch-cache/ONNX path policy and env overrides
│   ├── services/
│   │   ├── benchmarks.py            # PyTorch vs ONNX benchmarking, auto-export, busy flag
│   │   ├── cache_service.py         # Redis + in-memory LRU, versioned JSON serialisation
│   │   ├── diagnostics.py           # Module importability, ONNX weight inventory
│   │   ├── ground_truth.py          # GT decode, nearest-neighbour resize, median-scale align, metrics
│   │   ├── inference.py             # Public façade/orchestrator for image-to-depth responses
│   │   ├── inference_types.py       # Shared typed payload metadata for inference internals
│   │   ├── image_io.py              # Decode, depth normalization, colorization, PNG base64 encoding
│   │   ├── inference_cache.py       # Stable inference/depth cache keys and in-memory depth cache
│   │   ├── model_assets.py          # Runtime model asset readiness and remediation helpers
│   │   ├── model_runtime.py         # PyTorch MiDaS model/transform caches, locks, inference
│   │   ├── object_detection.py      # TorchVision RGB detection weight checks and inference
│   │   ├── observability.py         # Local-only counters, timings, and bounded histories
│   │   ├── onnx_inference.py        # ONNX engine cache, locks, inference, PyTorch fallback metadata
│   │   ├── metrics.py               # Metrics mode parsing, fast/full metrics, grouped payloads
│   │   ├── onnx_diagnostics.py      # ONNX session creation, provider selection, checker validation
│   │   └── reconstruction.py        # Pinhole projection, PLY/OBJ serialisation, preview downsampling
│   ├── scripts/
│   │   └── export_onnx.py           # ONNX export (torch.onnx + dynamo strategies), quarantine, validation
│   ├── tests/                       # Pytest suite — no GPU, Redis, or real model weights required
│   ├── utils/
│   │   └── hardware.py              # Device discovery, ONNX provider mapping, acceleration probe
│   ├── app.py                       # ASGI compatibility entry point (adds repo root to sys.path)
│   ├── constants.py                 # Shared lightweight literals for upload, modes, ONNX IDs, resource modes
│   ├── config.py                    # Pydantic settings with dotenv fallback
│   ├── depth_models.py              # ONNXExecutionEngine and DepthEstimator compatibility imports
│   ├── main.py                      # FastAPI app factory, CORS, JSON logging, lifespan
│   ├── model_metadata.py            # Lightweight aliases for backwards-compatible imports
│   ├── model_registry.py            # Canonical model specs, alias normalisation, ONNX path resolution
│   └── requirements.txt
│
├── electron-app/
│   ├── assets/                      # App icons for all platforms
│   ├── scripts/                     # Packaging, verification, lifecycle helpers
│   ├── src/
│   │   ├── main/                    # Focused main-process modules for desktop lifecycle and backend control
│   │   │   ├── backend-http.js       # /live JSON probes and DepthLens backend detection
│   │   │   ├── backend-lifecycle.js  # Backend startup, missing-resource errors, stale cleanup, safe shutdown
│   │   │   ├── backend-pid-store.js  # Private PID and backend metadata files
│   │   │   ├── paths.js              # App-root/resource/log path helpers
│   │   │   ├── ports.js              # Port availability and fallback discovery
│   │   │   ├── python-resolver.js    # Packaged/dev Python candidate selection
│   │   │   ├── settings-store.js     # Persisted settings schema, sanitization, corruption backups
│   │   │   └── windows.js            # Splash and main BrowserWindow options
│   │   ├── backend-process-policy.js # Backend ownership checks (command-line + PID metadata)
│   │   ├── platform-targets.js       # Central supported OS/architecture map, platform targets, settings bridge tests
│   │   └── security-policy.js        # Navigation allowlist, external URL classification
│   ├── main.js                      # Small Electron composition file — app lifecycle, IPC registration, module wiring
│   ├── preload.js                   # Narrow contextBridge surface
│   └── package.json
│
├── frontend/
│   ├── index.html                   # App shell and all workspace panels (Workspace, Webcam, Compare, Performance, Experiments, 3D, Guide)
│   ├── js/                          # Ordered browser scripts; ordered browser modules loaded directly by index.html
│   │   ├── state.js                 # Shared renderer state containers and defaults
│   │   ├── settings.js              # Theme, settings, localStorage, Electron settings bridge
│   │   ├── api-client.js            # Backend URL resolution, fetch wrappers, health/device/status calls
│   │   ├── dom.js                   # DOM lookup map and safe element helpers
│   │   ├── uploads.js               # File queue, validation, drag/drop
│   │   ├── inference-ui.js          # Estimate/batch progress and result rendering
│   │   ├── webcam.js                # Real-time camera inference lifecycle and cleanup
│   │   ├── compare.js               # Model comparison workspace
│   │   ├── benchmark.js             # PyTorch/ONNX benchmark panel
│   │   ├── experiments.js           # Experiment run/export workspace
│   │   ├── reconstruction.js        # 3D reconstruction and point-cloud preview
│   │   ├── charts.js                # Chart.js setup and theme handling
│   │   ├── notifications.js         # Settings panels, persistence hooks, toasts/navigation helpers
│   │   ├── observability.js         # Gallery/lightbox and observability dashboard
│   │   ├── performance.js           # Metrics dashboard refresh helpers
│   │   └── compat.js                # Final initialization, cleanup, utility compatibility
│   ├── script.js                    # Compatibility breadcrumb; index.html loads frontend/js directly
│   ├── style.css                    # Full design system (dark + light theme, CSS variables, animations)
│   └── welcome-anim.js              # Depth Field Calibration canvas animation (self-contained)
│
├── models/
│   └── onnx/                        # Generated ONNX files (not committed)
│
├── scripts/
│   ├── doctor.py                    # Cross-platform setup and environment verification
│   ├── diagnose_backend.py          # Port/process/endpoint diagnostics
│   ├── prefetch-detector-weights.py # Optional detector-weight cache prefetch
│   ├── prefetch-midas-assets.py     # MiDaS Torch Hub/checkpoint cache prefetch
│   ├── setup_state.py               # Shared setup report/status helpers
│   ├── setup-macos.sh
│   ├── setup-linux.sh
│   ├── setup-windows.ps1
│   ├── build-native-macos.sh
│   ├── build-native-linux.sh
│   └── build-native-windows.ps1
│
├── docs/
│   ├── debugging.md                 # Startup, asset, ONNX, port, packaged-resource, and settings troubleshooting
│   └── maintenance.md               # Safe extension guide for models, routes, installer/build, frontend modules, tests
│
├── .github/
│   └── workflows/ci.yml             # GitHub Actions — lint, type-check, pytest, Electron tests, resource dry-run
├── Dockerfile                       # Two-stage Python 3.12 slim build
├── docker-compose.yml
├── package.json                     # Root npm scripts
├── pyproject.toml                   # Black, Ruff, pytest configuration
├── mypy.ini
├── LICENSE
└── README.md
```

</details>

### Backend route organization

The inference service `backend/services/inference.py` preserves the existing public imports and response payloads delegates image I/O, cache keys, PyTorch runtime, ONNX fallback handling, and metrics to focused sibling modules.

The FastAPI route layer stays thin: public route declarations and high-level orchestration remain in `backend/api/routes.py`, while reusable validation, error mapping, device/readiness cache, and health telemetry helpers live in the neighboring `backend/api/` modules listed above. Public API routes, request fields, status codes, and response shapes are documented in the API Reference.

### Centralized path, platform, and model policy

DepthLens Pro keeps low-risk shared literals and path resolution in small central modules so setup, backend runtime, and packaging checks can stay in sync without changing public commands or API payloads. Backend upload/mode/model constants live in `backend/constants.py`, filesystem roots and ONNX environment override priority live in `backend/core/paths.py`, and Electron platform support remains centralized in `electron-app/src/platform-targets.js`. Standard builds still treat ONNX files as optional, while ONNX builds continue to require all supported ONNX model files.

### Reliability/performance hardening

Runtime hardening reduces avoidable work: Estimate cache hits now reuse normalized output metadata instead of reparsing it for telemetry, local observability hooks are failure-safe around inference paths, ONNX provider/device metadata is normalized before selection without changing provider priority, and Electron startup diagnostics keep a bounded backend-output tail with setup-remediation context.

---

Maintainer documentation lives in [`docs/maintenance.md`](docs/maintenance.md) and [`docs/debugging.md`](docs/debugging.md).

## Contributing

Contributions are welcome. Before opening a pull request, run the full check suite:

```bash
black --check .
ruff check .
mypy backend/
pytest

cd electron-app
npm test
cd ..
```

### PR Checklist

- Keep the change focused on one concern
- Add or update tests for all behaviour changes; aim for the same coverage the existing suite achieves without real GPU or model downloads
- Preserve existing API response shapes unless a breaking change is explicitly discussed first
- Update this README and relevant docs when setup, routes, runtime behaviour, or security properties change
- Avoid unrelated formatting-only changes in the same PR
- Run packaged-resource verification when touching Electron packaging scripts

### Code Style

- Python is formatted with Black (`line-length = 100`) and linted with Ruff (`E`, `F`, `W`, `I` rules)
- Type annotations are checked with mypy in strict mode (excluding `backend/tests/`)
- JavaScript follows the style already present in `main.js` and the ordered `frontend/js/` browser scripts — no separate formatter is enforced, but the Electron test suite (`npm test`) must pass

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## License

DepthLens Pro is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for the full terms.

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

## Acknowledgements

DepthLens Pro builds on excellent open-source projects:

| Project | Role |
|---|---|
| [Intel ISL MiDaS](https://github.com/isl-org/MiDaS) | MiDaS/DPT monocular depth estimation models |
| [PyTorch](https://pytorch.org) | Primary ML runtime and Torch Hub model loading |
| [ONNX Runtime](https://onnxruntime.ai) | Optional accelerated inference across CPU, CUDA, CoreML, OpenVINO |
| [FastAPI](https://fastapi.tiangolo.com) | Local HTTP API with async support and automatic OpenAPI docs |
| [Electron](https://www.electronjs.org) | Desktop application shell with context isolation |
| [OpenCV](https://opencv.org) | Image decoding, resizing, colourisation, and GT alignment |
| [NumPy](https://numpy.org) | Depth array arithmetic, GT metric computation |
| [Pillow](https://python-pillow.org) | PNG/TIFF/NPY GT file decoding |
| [Redis](https://redis.io) | Optional distributed cache backend |
| [Chart.js](https://www.chartjs.org) | Latency and benchmark charts in the frontend |

---

<div align="right"><sub><a href="#depthlens-pro">⬆ back to top</a></sub></div>

---

<div align="center">

**Made with care by [Ayushman Raha](https://github.com/AyushmanRaha)**

`com.ayushmanraha.depthlens-pro`

</div>

### Packaged startup readiness model

Packaged native apps start the local backend by waiting only for `GET /live`. `/live` is intentionally lightweight: it confirms that the FastAPI process is reachable and does not import Torch, OpenCV, ONNX Runtime, scan model caches, load ONNX graphs, or run CoreML provider checks. The renderer opens as soon as `/live` is healthy.

`GET /ready`, `GET /health`, and `GET /onnx/status` are deeper diagnostics. They default to `depth=quick`, use short-lived cached ONNX status results, and avoid opening every ONNX graph during UI startup. Use `depth=deep&force=true` from the Refresh flow or a terminal when you explicitly want a fresh ONNX Runtime session/checker validation. ONNX/CoreML diagnostics can take longer on first run; while they are pending the UI reports the engine as live/checking diagnostics rather than offline.

Standard native builds keep ONNX optional and verify the Python runtime, backend import path, Torch cache, and packaged resource layout:

```bash
npm run setup:mac
npm run verify:resources
npm run build:mac:arm64
npm run launch:mac

npm run setup:win
npm run build:win:arm64
npm run build:win:x64

npm run setup:linux
npm run build:linux:arm64
npm run build:linux:x64
```

ONNX native builds require all requested ONNX files and validate the same launchability checks after packaging:

```bash
npm run setup:mac:onnx
npm run verify:onnx:required
npm run build:mac:arm64:onnx
npm run launch:mac

npm run setup:win:onnx
npm run build:win:arm64:onnx
npm run build:win:x64:onnx

npm run setup:linux:onnx
npm run build:linux:arm64:onnx
npm run build:linux:x64:onnx
```

macOS x64 and universal native packages remain unsupported; use Apple Silicon arm64. To avoid launching a stale installed copy, open the freshly built artifact under `electron-app/dist/` (for macOS: `electron-app/dist/mac-arm64/DepthLens Pro.app`) or run the repository launch command after the build. Electron startup logs include the resource root, Python path, backend directory, models and ONNX directories, `app.isPackaged`, executable path, `process.resourcesPath`, current working directory, and backend URL. Check the Electron log path shown in backend startup errors and the backend stdout/stderr tail included in those errors.

If the UI says “Depth engine offline,” first verify liveness:

```bash
curl http://127.0.0.1:8765/live
curl 'http://127.0.0.1:8765/ready?depth=quick'
curl 'http://127.0.0.1:8765/onnx/status?depth=deep&force=true'
```

If `/live` succeeds, the backend is not offline; diagnostics may still be pending or degraded. If `/live` fails, inspect the Electron logs for a port conflict on 8765, a fallback backend URL, stale PID metadata, missing packaged resources, or a stale installed app path. If another non-DepthLens process owns 8765, stop that process or set `DEPTHLENS_BACKEND_PORT` before launch.
