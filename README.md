# DepthLensPro

[![CI/CD Status](https://img.shields.io/badge/CI%2FCD-passing-placeholder.svg)](#validation--ci)
[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](electron-app/package.json)
[![API](https://img.shields.io/badge/API-3.1.0-6f42c1.svg)](backend/api/live.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Platform](https://img.shields.io/badge/platform-ARM%20desktop%20only-lightgrey.svg)](#supported-platforms)

Last Updated: 11th June 2026

DepthLensPro is a local-first desktop application for hardware-aware monocular depth estimation from 2D images. It combines an Electron `4.0.0` shell with a FastAPI `3.1.0` backend that runs MiDaS-family models through PyTorch and ONNX Runtime, then presents depth maps, diagnostics, experiments, and performance analysis in one secure desktop workspace.

The project favors explicit runtime checks, small reviewable changes, ARM desktop packaging, repeatable dependency installation, and safe fallback behavior when optional accelerators or Redis are unavailable.

---

## Table of Contents

- [Capability Overview](#capability-overview)
- [Hardware Acceleration](#hardware-acceleration)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Observability & Telemetry](#observability--telemetry)
- [Quick Start / Runbook](#quick-start--runbook)
- [Supported Platforms](#supported-platforms)
- [API Surface](#api-surface)
- [Metrics & Models](#metrics--models)
- [Project Structure](#project-structure)
- [Validation & CI](#validation--ci)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Capability Overview

### Core Engine & AI

- Monocular depth estimation with canonical model IDs `midas_small`, `dpt_hybrid`, and `dpt_large`; display names are `MiDaS Small`, `DPT Hybrid`, and `DPT Large`.
- Alias-aware model input: the backend accepts common display-name, PyTorch-name, space, hyphen, and underscore variants, then normalizes them to lowercase canonical IDs internally.
- PyTorch inference with ONNX Runtime acceleration when exported static `.onnx` graphs and compatible providers are available.
- Single-image and batch inference, with batch requests capped at **10 images**.
- Supported source uploads: FastAPI `image/*` uploads that OpenCV can decode, with a **20 MB** route limit; `/batch` rejects non-image items before inference.
- Optional Ground Truth mode for one source image plus one GT depth file, with `.png`, `.tif`/`.tiff`, and `.npy` support.
- Selective output generation for colorized and/or grayscale depth maps.
- Colormap options: `inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, and `turbo`.
- Runtime model comparison, Performance Analysis, session analytics, and Experiment Workspace exports.
- Structured metrics that separate prediction statistics, proxy/self-consistency diagnostics, GT metrics, and unavailable metrics.

### System Architecture & Observability

- FastAPI ASGI backend with separate route, live, inference, cache, benchmark, diagnostics, configuration, hardware, and model-registry modules.
- Redis-backed response cache with in-memory fallback, plus a lock-protected normalized-depth in-process cache keyed by image, model, device, and max dimension.
- Versioned JSON cache serialization with schema version `2` and magic prefix `DLP2\0`; legacy pickle payloads using `DLP1\0` or raw pickle bytes return cache misses and are not deserialized.
- Environment-backed configuration through process variables and optional `.env` values.
- JSON structured logging for backend runtime logs.
- Lightweight liveness, inference readiness, ONNX status, device inventory, health diagnostics, benchmark, cache, memory, disk, and acceleration telemetry endpoints.
- Sanitized structured API error envelopes for dependency and inference failures; detailed exception text stays in backend logs.
- Docker and Docker Compose support for backend-plus-Redis execution.

### Desktop Client

- Electron desktop shell with automatic FastAPI child-process lifecycle management.
- Sandboxed renderer with `contextIsolation` enabled, `nodeIntegration` disabled, and strict navigation policy.
- Startup sequence: Electron waits on `/live`, the renderer calls `/ready` before enabling inference controls, then `/health`, `/devices`, and `/cache/metrics` load background diagnostics.
- Drag-and-drop upload workflow with progress, ETA, cancellation, optional GT pairing, result cards, lightbox metrics, and download actions.
- Persistent preferences for model, colormap, theme, and compute device.
- Workspace panels for generation, model comparison, performance analysis, experiments, session analytics, and application details.
- Animated landing experience with theme-aware visuals.

---

## Hardware Acceleration

DepthLensPro treats accelerator selection as a runtime setting. Users can choose `auto`, `cpu`, `cuda`, `mps`, or `xpu`; the backend validates availability and falls back safely when a requested provider cannot run.

| Target | Runtime Path | Notes |
|---|---|---|
| **Apple Silicon** | PyTorch MPS via Metal | Native `darwin arm64` packaging avoids Rosetta for the desktop app and PyTorch path. |
| **NVIDIA GPU** | PyTorch CUDA and matching ONNX Runtime providers where installed | Best for larger DPT models and repeated workloads. |
| **Intel XPU** | PyTorch XPU where available | Exposed as a selectable device when detected by the hardware utility. |
| **CPU** | PyTorch CPU and ONNX Runtime CPU provider | Portable fallback for all supported systems. |
| **ONNX Runtime** | Static `.onnx` graphs with provider filtering | Uses exported graphs from the configured ONNX search path and reports provider fallback in diagnostics. |

Desktop packaging targets ARM platforms only. The backend can still expose CPU, CUDA, MPS, XPU, and ONNX provider data according to the Python runtime installed on that machine.

---

## Architecture

DepthLensPro uses a desktop-first architecture: Electron owns the window, process lifecycle, and secure bridge; FastAPI owns inference and diagnostics; the frontend owns user workflow and visualization.

```text
DepthLensPro/
├── electron-app/          # Electron shell, lifecycle, packaging, security policy
├── frontend/              # Static renderer workspace
├── backend/               # FastAPI API, inference, diagnostics, cache, models
├── models/onnx/           # Repository-default ONNX graph directory
├── Dockerfile             # Backend container image
├── docker-compose.yml     # Backend + Redis development stack
└── README.md              # Operator and developer runbook
```

```text
Electron main process
  ├─ checks supported ARM desktop architecture
  ├─ starts uvicorn backend on 127.0.0.1:8765 unless a live owned backend exists
  ├─ waits for GET /live
  ├─ loads frontend/index.html through file://
  └─ exposes a narrow preload bridge

Renderer
  ├─ resolves backend URL from preload
  ├─ calls GET /live
  ├─ calls GET /ready before enabling inference controls
  ├─ loads GET /health, GET /devices, and GET /cache/metrics in the background
  └─ calls inference, batch, benchmark, cache, and ONNX endpoints as needed

FastAPI backend
  ├─ validates uploads, model IDs, colormaps, metrics mode, outputs, and device requests
  ├─ normalizes model aliases to canonical lowercase IDs
  ├─ executes PyTorch or ONNX Runtime paths with PyTorch fallback
  ├─ computes metrics and optional GT alignment
  └─ returns structured diagnostics for UI and operators
```

### Separation of Concerns

| Layer | Responsibility | Key Files |
|---|---|---|
| Electron main | App lifecycle, backend child process, supported-architecture guard, navigation policy, IPC handlers | `electron-app/main.js`, `electron-app/src/security-policy.js`, `electron-app/src/backend-process-policy.js` |
| Preload bridge | Minimal renderer API surface | `electron-app/preload.js` |
| Renderer | Workspace UI, preferences, polling, uploads, comparisons, experiments, exports | `frontend/index.html`, `frontend/script.js` |
| API routes | HTTP request validation, endpoint payloads, route-level cache use | `backend/api/live.py`, `backend/api/routes.py` |
| Services | Inference, benchmarks, cache, diagnostics, GT metrics, ONNX diagnostics | `backend/services/` |
| Models | Registry, alias normalization, ONNX path resolution, PyTorch/ONNX wrappers | `backend/model_registry.py`, `backend/depth_models.py` |

---

## Technology Stack

Version values come from `backend/requirements.txt` and `electron-app/package.json`.

| Component | Version | Role |
|---|---:|---|
| DepthLens Pro app | `4.0.0` | Electron application version. |
| DepthLens Pro API | `3.1.0` | FastAPI service version. |
| Electron | `^42.3.0` | Native desktop shell. |
| electron-builder | `^26.8.1` | ARM desktop packaging. |
| FastAPI | `0.135.3` | ASGI API framework. |
| Uvicorn | `0.44.0` | ASGI server launched by Electron and Docker. |
| PyTorch | `2.11.0` | Primary model runtime and ONNX export source. |
| torchvision | `0.26.0` | PyTorch vision dependency. |
| ONNX | `1.20.1` | Exported graph format and validation library. |
| ONNX Runtime | `1.24.3` | Optional static-graph inference runtime. |
| onnxscript | `0.7.0` | PyTorch ONNX export support. |
| timm | `1.0.26` | Model dependency used by MiDaS-family loading. |
| OpenCV Headless | `4.13.0.92` | Image decoding, resizing, colorization, and metrics operations. |
| NumPy | `2.4.4` | Array and metric computation. |
| Pillow | `12.2.0` | GT image decoding. |
| Redis Python client | `6.4.0` | Optional Redis cache backend. |
| pydantic-settings | `2.12.0` | Environment-backed configuration. |
| python-multipart | `0.0.24` | FastAPI multipart upload parsing. |
| Docker Compose services | `api`, `redis` | Backend and optional cache service for containerized development. |

---

## Observability & Telemetry

The backend exposes cheap liveness separately from heavier diagnostics so the app can start quickly while detailed checks load in the background.

| Signal | Endpoint or Surface | Description |
|---|---|---|
| Liveness | `GET /live` | Lightweight process heartbeat with service name, API version, PID, timestamp, and uptime. |
| Inference readiness | `GET /ready` | Required Python module checks, optional module checks, torch runtime details, ONNX status, models, colormaps, and effective settings. |
| Health diagnostics | `GET /health` | Device inventory, loaded models, cache metrics, PyTorch version, acceleration probes, ONNX readiness, memory, disk, warmup, timings, and system metadata. |
| Device inventory | `GET /devices` | Cached device list and primary device; always falls back to CPU if discovery fails. |
| ONNX diagnostics | `GET /onnx/status` | Supported model IDs, expected and selected paths, file sizes, runtime import status, available providers, selected providers, and export commands. |
| Cache metrics | `GET /cache/metrics`, `GET /health` | Cache backend, Redis availability, hits, misses, keyspace size, failures, TTL, and memory limit. |
| Runtime benchmarks | `GET /benchmark`, `GET /api/benchmark` | PyTorch versus ONNX Runtime latency, throughput, memory, provider, speedup, and diagnostic state. |
| Session analytics | Desktop workspace | Processed image count, latency history, min/max/average latency, throughput, total inference time, cache hits, and errors. |
| Experiment validation | Desktop Experiments panel | Named local runs, image/GT metadata, latency, metrics, warnings, previews, error heatmaps, and JSON/CSV export. |

---

## Supported Platforms

DepthLens Pro supports ARM desktop targets for native app packaging. Normal users should open the native app; Electron starts and stops FastAPI for them.

| Status | Platform | Architecture | Notes |
|---|---|---|---|
| Supported | macOS Apple Silicon only | `darwin arm64` | Build/open `electron-app/dist/mac-arm64/DepthLens Pro.app` or install the Apple Silicon DMG. |
| Supported | Windows ARM only | `win32 arm64` | Build with `npm run build:win:arm64`; x64 installers are not produced. |
| Supported | Linux ARM only | `linux arm64` / `aarch64` | Build with `npm run build:linux:arm64`; x64 AppImages are not produced. |
| Unsupported | Intel Mac / macOS x64 | `darwin x64` | Blocked at Electron startup before backend launch. |
| Unsupported | macOS universal builds | `universal` | Unsupported because they can create duplicate or stale bundles. |
| Unsupported | Windows x64 | `win32 x64` | Unsupported build scripts fail with a clear architecture message. |
| Unsupported | Linux x64 | `linux x64` | Unsupported build scripts fail with a clear architecture message. |

The bundle identifier is `com.ayushmanraha.depthlens-pro`, and the product name is `DepthLens Pro`.

---

## Quick Start / Runbook

DepthLensPro uses a local FastAPI backend at `http://127.0.0.1:8765` by default. Electron starts that backend automatically for the native app; terminal-only development can start the same backend manually.

### Section A — Prerequisites

**macOS Apple Silicon prerequisites:** Install Git, Node.js + npm from nodejs.org, and Python 3.12 from python.org (recommended) or Homebrew. Do **not** use a bare `python3 -m venv`; use the setup scripts below so DepthLens Pro selects and validates the correct interpreter.

**Windows ARM prerequisites:** Install Git, Node.js + npm, and Python 3.12 for ARM64 from python.org with the `py` launcher enabled.

**Linux ARM prerequisites:** Install Git, Node.js + npm, and `python3.12` + `python3.12-venv` from your distro package manager.

### Section B — Native app build

#### macOS Apple Silicon

1. Clone the repository.

   ```bash
   git clone https://github.com/AyushmanRaha/DepthLensPro.git
   cd DepthLensPro
   ```

2. Run the one-command setup + native build.

   ```bash
   scripts/build-native-macos.sh
   ```

   This runs the doctor, creates the venv with Python 3.12, installs backend deps, npm installs, cleans stale dist, verifies repo-root resources, builds the `.app`, and verifies the packaged `Contents/Resources` tree before the command succeeds.

3. Open the app.

   ```bash
   open "electron-app/dist/mac-arm64/DepthLens Pro.app"
   ```

#### Windows ARM

1. Clone the repository.

   ```powershell
   git clone https://github.com/AyushmanRaha/DepthLensPro.git
   cd DepthLensPro
   ```

2. Run in PowerShell:

   ```powershell
   .\scripts\build-native-windows.ps1
   ```

3. Install the generated NSIS installer from `electron-app/dist/`. The build script verifies repo-root resources before packaging and the generated Windows ARM unpacked resource tree after packaging before the installer is considered valid.

#### Linux ARM

1. Clone the repository.

   ```bash
   git clone https://github.com/AyushmanRaha/DepthLensPro.git
   cd DepthLensPro
   ```

2. Run:

   ```bash
   scripts/build-native-linux.sh
   ```

3. Run the AppImage. The build script verifies repo-root resources before packaging and verifies the Linux ARM unpacked resource tree that electron-builder leaves in `electron-app/dist` before the command succeeds:

   ```bash
   ./electron-app/dist/DepthLens\ Pro-*-linux-arm64.AppImage
   ```

### Section C — Terminal-only local development (no Electron, backend only)

1. Run setup (creates venv, installs deps):

   ```bash
   npm run setup          # macOS/Linux
   npm run setup:win      # Windows PowerShell
   ```

2. Start the FastAPI backend:

   ```bash
   npm run backend:dev
   ```

3. In a second terminal, start the Electron dev shell:

   ```bash
   npm run frontend:dev
   ```

4. Verify the backend is live:

   ```bash
   curl http://127.0.0.1:8765/live
   ```

### Section D — Troubleshooting

**Symptom:** "pip install failed" with CalledProcessError but no visible pip output. **Cause:** Python 3.10 was selected but the pinned deps require Python 3.11+. **Fix:** Install Python 3.12 from python.org, delete the venv (`rm -rf venv`), and re-run the setup script.

**Symptom:** App opens but shows "Required app resources were not found" / `models/onnx` missing. **Cause:** The packaged app is incomplete or you are launching a stale installed app. `models/` and `models/onnx/` are required packaged runtime directories even when actual `.onnx` model files are optional. **Fix:** Re-run the supported root native build script. If you installed or copied an older app, remove `/Applications/DepthLens Pro.app` on macOS, uninstall/reinstall the Windows app, or replace the old Linux desktop/AppImage location before launching the newly built artifact.

**Symptom:** `zsh: command not found: #` when pasting commands. **Cause:** zsh interactive comments are disabled. **Fix:** Run `setopt interactivecomments` in your terminal session, or paste commands without comment lines.

**Symptom:** `/live` returns `ok`, but inference controls stay disabled. **Cause:** `/ready` found a missing required Python module such as `torch`, `cv2`, `PIL`, `fastapi`, `uvicorn`, or `numpy`. **Fix:** Activate the repo-root virtual environment and run `python -m pip install -r backend/requirements.txt && venv/bin/python -m pip check`.

**Symptom:** MiDaS Small ONNX Performance Analysis is unavailable. **Cause:** ONNX acceleration is optional and the ONNX file may not have been exported. **Fix:** Run `venv/bin/python backend/scripts/export_onnx.py --model midas_small --force`; PyTorch fallback remains available.

**Symptom:** DPT Hybrid or DPT Large ONNX is unavailable. **Cause:** Those exports are large and provider/tooling-sensitive. **Fix:** Treat them as optional unless `--require-all` is passed; PyTorch fallback remains available and diagnostics identify the precise validation failure.

## API Surface

All endpoints are served by the FastAPI app mounted from `backend.api.live` and `backend.api.routes`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Returns service name and API version. |
| `GET` | `/live` | Returns lightweight liveness payload with `status`, `service`, `version`, `pid`, `timestamp`, `uptime_seconds`, and optional busy state. |
| `GET` | `/ready` | Reports whether required inference dependencies are importable without loading models, plus optional modules, torch runtime, ONNX weights, models, colormaps, settings, and duration. |
| `GET` | `/health` | Returns full diagnostics: status, version, primary device, devices, loaded models, cache entries and metrics, torch version, acceleration flags and checks, ONNX diagnostics, readiness, warmup, timings, telemetry, and system metadata. |
| `GET` | `/devices` | Returns cached device inventory, primary device, and whether the payload came from the device cache. |
| `GET` | `/models` | Returns registry model specs with canonical IDs, display names, architecture, input size, PyTorch names, preprocessing, ONNX filenames, support flags, recommended device, and notes. |
| `GET` | `/colormaps` | Returns the supported colormap names. |
| `POST` | `/estimate` | Accepts one image upload and form fields for `model`, `colormap`, `device`, `metrics`, `outputs`, `max_dim`, and optional GT fields; returns depth outputs, metrics, model/device metadata, cache flags, and GT visualizations when supplied. Generic inference failures return a sanitized `INFERENCE_FAILED` error envelope. |
| `POST` | `/batch` | Accepts up to 10 image uploads with common form options and returns `results`, `errors`, `total`, `succeeded`, and `failed`; non-image items are rejected per file before inference. |
| `GET` | `/benchmark` | Runs PyTorch versus ONNX Runtime benchmark matrices for `model`, `device`, and `iterations`. Missing ONNX weights are reported as unavailable unless `DEPTHLENS_AUTO_EXPORT_ONNX=true` explicitly enables request-time export. |
| `GET` | `/api/benchmark` | Alias for `/benchmark`, used by the frontend Performance panel. |
| `GET` | `/cache/metrics` | Returns live cache backend, Redis availability, hits, misses, keyspace, Redis failures, TTL, and memory limits. |
| `DELETE` | `/cache` | Clears Redis keys and in-memory cache entries, then returns the number of cleared entries. |
| `GET` | `/onnx/status` | Returns ONNX runtime import/provider diagnostics and per-model path/session status for the requested device. |

The `/estimate` form accepts `model` (`str`, default `MiDaS_small`, normalized to canonical IDs), `colormap` (`str`, default `inferno`), `device` (`str`, default `auto`), `metrics` (`str`, default from `DEPTHLENS_DEFAULT_METRICS`; `DEPTHLENS_DEFAULT_METRICS` is `str`, default `fast`), `outputs` (`str`, default from `DEPTHLENS_DEFAULT_OUTPUTS`; `DEPTHLENS_DEFAULT_OUTPUTS` is `str`, default `color`), `max_dim` (`int | None`, default `None`; when omitted, `DEPTHLENS_MAX_DIM` is `int`, default `1536`), `gt_required` (`bool`, default `False`), `gt_scale` (`float | None`, default `None`), and `gt_invalid_value` (`float | None`, default `None`).

<details>
<summary><strong>Example <code>/health</code> response</strong></summary>

```json
{
  "status": "ok",
  "diagnostics_status": "ok",
  "version": "3.1.0",
  "primary_device": "cpu",
  "devices": {
    "cpu": {
      "name": "CPU",
      "type": "cpu",
      "available": true
    }
  },
  "loaded_models": [],
  "cache_entries": 0,
  "cache_metrics": {
    "backend": "memory",
    "redis_available": false,
    "total_hits": 0,
    "cache_misses": 0,
    "keyspace_size": 0,
    "memory_hits": 0,
    "memory_misses": 0,
    "memory_keyspace_size": 0,
    "redis_failures": 0,
    "ttl_seconds": 3600,
    "memory_max_entries": 256
  },
  "torch_version": "2.11.0",
  "cuda_available": false,
  "mps_available": false,
  "xpu_available": false,
  "acceleration_ok": true,
  "acceleration_checks": {},
  "onnx": {
    "supported_model_ids": ["midas_small", "dpt_hybrid", "dpt_large"],
    "requested_device": "cpu",
    "overall_status": "onnx_ready"
  },
  "readiness": {
    "backend_live": true,
    "overall_status": "onnx_ready",
    "warnings": []
  },
  "backend_live": true,
  "overall_status": "onnx_ready",
  "model_readiness": {},
  "warmup": {
    "enabled": false,
    "model": "MiDaS_small",
    "device": "auto",
    "loaded_models": []
  },
  "timings_ms": {
    "device_discovery": 0.5,
    "accelerator_probe": 0.2,
    "cache_metrics": 0.1,
    "readiness": 0.4,
    "health_generation": 1.5
  },
  "telemetry": {
    "memory": {
      "status": "ok",
      "pressure_percent": 42.0,
      "limit_percent": 90.0
    },
    "disk": {
      "status": "ok",
      "path": "/",
      "usage_percent": 55.0,
      "limit_percent": 90.0
    }
  },
  "system": {
    "os": "Linux-6.x-aarch64-with-glibc2.x",
    "machine": "aarch64",
    "cpu": "System CPU",
    "accelerators": []
  }
}
```

</details>

<details>
<summary><strong>Example <code>/benchmark</code> response</strong></summary>

```json
{
  "model": "midas_small",
  "model_id": "midas_small",
  "display_name": "MiDaS Small",
  "input_shape": [1, 3, 384, 384],
  "device_requested": "auto",
  "device_resolved": "cpu",
  "iterations": 3,
  "frame_shape": [384, 384, 3],
  "weights": {
    "onnx_path": "/workspace/DepthLensPro/models/onnx/midas_small.onnx",
    "onnx_available": true
  },
  "onnx_diagnostics": {
    "model": "midas_small",
    "display_name": "MiDaS Small",
    "state": "available"
  },
  "results": [
    {
      "engine": "pytorch",
      "status": "ok",
      "state": "available",
      "iterations": 3,
      "latency_ms": { "avg": 120.0, "min": 110.0, "max": 130.0, "samples": [110.0, 120.0, 130.0] },
      "throughput_fps": 8.33,
      "throughput_frames_per_min": 500.0,
      "memory": { "before": { "process_rss_mb": 500.0 }, "after": { "process_rss_mb": 505.0 }, "process_rss_delta_mb": 5.0 }
    },
    {
      "engine": "onnxruntime",
      "status": "ok",
      "state": "available",
      "iterations": 3,
      "latency_ms": { "avg": 80.0, "min": 75.0, "max": 85.0, "samples": [75.0, 80.0, 85.0] },
      "throughput_fps": 12.5,
      "throughput_frames_per_min": 750.0,
      "memory": { "before": { "process_rss_mb": 505.0 }, "after": { "process_rss_mb": 506.0 }, "process_rss_delta_mb": 1.0 },
      "provider": "CPUExecutionProvider",
      "providers": ["CPUExecutionProvider"],
      "uses_cpu_fallback": false
    }
  ],
  "comparison": { "latency_delta_ms": 40.0, "speedup": 1.5, "faster_engine": "onnxruntime" },
  "speedup": 1.5,
  "pytorch": { "status": "ok", "latency_ms": 120.0, "throughput_fps": 8.33, "device_used": "cpu", "memory_mb": 506.0, "error": null },
  "onnx": { "status": "ok", "latency_ms": 80.0, "throughput_fps": 12.5, "providers_used": ["CPUExecutionProvider"], "onnx_path": "/workspace/DepthLensPro/models/onnx/midas_small.onnx", "error_code": null, "message": null, "technical_detail": null },
  "warnings": []
}
```

</details>

<details>
<summary><strong>Example <code>/estimate</code> response</strong></summary>

```json
{
  "depth_map": "iVBORw0KGgo...",
  "metrics": {
    "min": 0.0,
    "max": 1.0,
    "mean": 0.4821,
    "std": 0.1822,
    "entropy": 4.317,
    "histogram": { "counts": [0, 4, 12], "bin_edges": [0.0, 0.031, 0.063] },
    "prediction_stats": {
      "min": 0.0,
      "max": 1.0,
      "mean": 0.4821,
      "std": 0.1822,
      "entropy": 4.317,
      "histogram": { "counts": [0, 4, 12], "bin_edges": [0.0, 0.031, 0.063] }
    },
    "proxy_metrics": {},
    "gt_metrics": {},
    "unavailable": {
      "abs_rel": "needs_gt_depth_upload",
      "delta_1": "needs_gt_depth_upload",
      "mae": "not_requested_fast_mode",
      "rmse": "not_requested_fast_mode",
      "ssim": "not_requested_fast_mode"
    },
    "warnings": []
  },
  "latency_ms": 95.4,
  "model": "midas_small",
  "model_id": "midas_small",
  "model_display_name": "MiDaS Small",
  "colormap": "inferno",
  "device_used": "cpu",
  "resolution": { "width": 1024, "height": 768 },
  "filename": "sample.png",
  "cached": false,
  "depth_cached": false,
  "metrics_mode": "fast",
  "outputs": ["color"],
  "gt_metadata": { "provided": false },
  "engine_requested": "auto",
  "engine_used": "pytorch",
  "device_requested": "cpu",
  "fallback_used": false,
  "warnings": []
}
```

</details>

---

## Metrics & Models

DepthLensPro reports depth statistics, proxy diagnostics, and GT metrics in one response while keeping their meanings distinct. Use GT metrics for benchmark comparisons when valid labels are uploaded.

### Metrics Reference

The `/estimate` response returns `metrics` as a structured object with `prediction_stats`, `proxy_metrics`, `gt_metrics`, and `unavailable`. Flat metric keys remain present at the top of `metrics` for backward compatibility.

| Key | Group | Label | Meaning |
|---|---|---|---|
| `min`, `max`, `mean`, `std`, `median`, `dynamic_range`, `entropy`, `coverage`, `histogram` | `prediction_stats` | Prediction distribution | Descriptive statistics for the normalized predicted depth plane. |
| `ssim` | `proxy_metrics` | RGB–Depth Structural Proxy | Structural similarity between grayscale RGB and normalized depth; useful only as a self-consistency signal. |
| `silog` | `proxy_metrics` | Log-Depth Dispersion Proxy | Scale-invariant log dispersion within the predicted depth map. |
| `psnr` | `proxy_metrics` | Depth Variance PSNR Proxy | PSNR-like score derived from depth variance around the mean. |
| `gradient_error` | `proxy_metrics` | Depth Edge Proxy | Mean depth-gradient magnitude used as an edge proxy. |
| `mae`, `rmse`, `log_rmse`, `gradient_mean`, `gradient_std`, `edge_density` | `proxy_metrics` | Prediction proxy diagnostics | Self-consistency and depth-shape diagnostics without ground truth. |
| `abs_rel`, `sq_rel`, `gt_mae`, `gt_rmse`, `gt_log_rmse`, `delta_1`, `delta_2`, `delta_3` | `gt_metrics` | Ground-truth benchmark metrics | Metrics computed after GT validation, nearest-neighbor alignment, and median-scale alignment. |
| `gt_ssim`, `gt_psnr`, `ordinal_error`, `surface_normal_error`, `lpips` | `unavailable` | Not implemented | Explicitly marked unavailable rather than synthesized. |

`metrics=fast` returns lightweight prediction statistics and marks full-only metrics as `not_requested_fast_mode`. `metrics=full` computes prediction statistics and proxy diagnostics. `metrics=none` returns grouped availability metadata without expensive metric computation.

### Ground Truth Mode and Metric Alignment

Ground truth uploads support `.png`, `.tif`/`.tiff`, and `.npy` files up to **20 MB**. Multi-channel PNG/TIFF files convert to grayscale with a warning; multi-channel NPY files are rejected unless they have shape H×W×1. EXR and PFM are not supported.

GT depth is decoded as `float32`, filtered to finite positive pixels, optionally scaled by `gt_scale` (`float | None`, default `None`), optionally filtered by `gt_invalid_value` (`float | None`, default `None`), resized to prediction H×W with nearest-neighbor sampling, median-scale aligned to the relative prediction, and then evaluated. GT requests bypass response-cache reuse because uploaded labels change metric results.

### Experiment Workspace

The Experiments panel stores named local validation runs in the renderer. It shows RGB, prediction, GT, and error previews when GT is valid, summarizes metrics across runs, and exports JSON or CSV for offline analysis.

### Model Reference

| Canonical ID | Display Name | PyTorch Model Name | ONNX Filename | Input Size | Recommended Device |
|---|---|---|---|---:|---|
| `midas_small` | MiDaS Small | `MiDaS_small` | `midas_small.onnx` | `384×384` | CPU or GPU |
| `dpt_hybrid` | DPT Hybrid | `DPT_Hybrid` | `dpt_hybrid.onnx` | `384×384` | GPU preferred |
| `dpt_large` | DPT Large | `DPT_Large` | `dpt_large.onnx` | `384×384` | GPU recommended for speed |

The backend accepts aliases such as `MiDaS_small`, `MiDaS Small`, `midas-small`, `DPT_Hybrid`, `DPT Hybrid`, and `dpt-large`, then returns canonical lowercase IDs in API responses.

---

## Project Structure

```text
DepthLensPro/
├── AGENTS.md
├── CONTRIBUTING.md
├── Dockerfile
├── README.md
├── docker-compose.yml
├── mypy.ini
├── pyproject.toml
├── backend/
│   ├── api/
│   │   ├── live.py
│   │   └── routes.py
│   ├── services/
│   │   ├── benchmarks.py
│   │   ├── cache_service.py
│   │   ├── diagnostics.py
│   │   ├── ground_truth.py
│   │   ├── inference.py
│   │   └── onnx_diagnostics.py
│   ├── scripts/
│   │   └── export_onnx.py
│   ├── tests/
│   ├── config.py
│   ├── depth_models.py
│   ├── main.py
│   ├── model_metadata.py
│   ├── model_registry.py
│   └── requirements.txt
├── electron-app/
│   ├── main.js
│   ├── preload.js
│   ├── package.json
│   ├── scripts/
│   └── src/
├── frontend/
│   ├── index.html
│   └── script.js
└── models/
    ├── README.md
    └── onnx/
        └── README.md
```

---

## Validation & CI

CI and local checks focus on lightweight validation with mocks or stubs where model downloads, GPUs, browsers, Docker, and external services would make tests slow or environment-dependent.

| Check | Source | Purpose |
|---|---|---|
| Python tests | `backend/tests/` | Cache serialization safety, route behavior, ONNX diagnostics, benchmarks, warmup guards, GT handling, and mocked inference flows. |
| Python format/lint/type gates | `pyproject.toml`, `mypy.ini`, `.github/workflows/ci.yml` | Black, Ruff, mypy, and pytest configuration. |
| Electron resource verification | `electron-app/scripts/verify-resources.js` | Confirms repo-root or packaged resource paths for backend, frontend, platform venv Python, `models/`, and `models/onnx/`. |
| Packaged artifact verification | `electron-app/scripts/verify-packaged-resources.js` | Discovers electron-builder macOS, Windows ARM, and Linux ARM resource roots after packaging and fails incomplete native artifacts. |
| Electron security policy test | `electron-app/test-security-policy.js` | Validates navigation policy behavior. |
| Packaging scripts | `electron-app/package.json` | Builds ARM-only macOS, Windows, and Linux artifacts; unsupported x64 scripts call `unsupported-arch.js`. |

This command runs the normal Python validation gates in an already prepared environment.

```bash
black --check .
ruff check .
mypy backend/
pytest
```

This command verifies Electron repo-root resources from the Electron app directory.

```bash
cd electron-app
npm run verify:resources:native
```

These commands verify packaged output after a native build. They are also run by the supported root native build scripts and by the direct `electron-app` ARM build scripts.

```bash
cd electron-app
npm run verify:packaged:mac      # macOS Apple Silicon .app resources
npm run verify:packaged:win      # Windows ARM unpacked resources
npm run verify:packaged:linux    # Linux ARM unpacked resources
```

---

## Troubleshooting

Use the symptom, cause, and resolution entries below for common local issues. Keep `/live` separate from `/ready`: `/live` means the process answers, while `/ready` means required inference imports are available.

### Setup, Python, certificates, and ONNX packaging

**Symptom** → setup accidentally uses Python 3.14 or an old `venv` created by Python 3.14. **Cause** → `python3` can point at an unsupported interpreter and old virtualenvs can mask the problem. **Resolution** → run `scripts/setup-macos.sh`, `scripts/setup-linux.sh`, or `./scripts/setup-windows.ps1`; the doctor rejects Python <3.10 and >3.12, detects an unsupported existing `venv`, and recreates the project-owned venv safely.

**Symptom** → Homebrew Python fails during `ensurepip` or imports `pyexpat` incorrectly. **Cause** → the Python install itself is broken before DepthLens dependencies are installed. **Resolution** → the doctor probes `ensurepip`, `ssl`, `venv`, `pyexpat`, and `pip` first and prints remediation. On macOS, install Python 3.12 from python.org if Homebrew remains broken.

**Symptom** → Torch Hub or GitHub downloads fail with SSL certificate verification errors. **Cause** → Python cannot locate a current CA bundle. **Resolution** → setup installs/upgrades `certifi`; export-time code sets `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` from `certifi` when available.

**Symptom** → Torch Hub asks “Do you trust this repository?” during ONNX export. **Cause** → interactive Torch Hub trust prompts are unsafe for builds/CI. **Resolution** → `backend/scripts/export_onnx.py` passes Torch Hub trust options and is non-interactive by default.

**Symptom** → packaged Performance Analysis reports `ONNX missing_model_file`. **Cause** → no optional ONNX graph was exported for that model, or native resources did not include `models/onnx`. **Resolution** → `models/onnx/` is required as a directory in every native package, but `.onnx` binaries remain optional for the default build and PyTorch fallback remains available. Run `cd electron-app && npm run verify:resources:native` before building and `npm run verify:packaged:mac`, `npm run verify:packaged:win`, or `npm run verify:packaged:linux` after building.

**Symptom** → ONNX is reported as invalid/corrupt. **Cause** → the file may be empty, fail `onnx.checker`, fail ONNX Runtime session creation, fail dummy inference, or use a provider unavailable on the current system. **Resolution** → run `venv/bin/python backend/scripts/export_onnx.py --validate-only --all`; invalid final-path files are quarantined and diagnostics report `empty`, `invalid_checker`, `invalid_session`, `invalid_dummy_inference`, `provider_unavailable`, `runtime_unavailable`, `export_failed`, or `optional_unavailable` instead of one generic error.

**Symptom** → MiDaS Small ONNX expected 384 but runtime supplied 256. **Cause** → MiDaS `small_transform` produces 256×256 input. **Resolution** → registry/export metadata now uses `midas_small` input shape `1x3x256x256`; DPT Hybrid/Large remain `1x3x384x384`.

**Symptom** → DPT Hybrid or DPT Large ONNX is unavailable. **Cause** → those exports are large and provider/tooling-sensitive. **Resolution** → they are optional unless `--require-all` is passed; PyTorch fallback remains available and Performance Analysis displays the precise diagnostic and recommended command.

**Symptom** → backend appears offline during a long benchmark. **Cause** → model loading or CPU-heavy inference can take longer than UI polling timeouts. **Resolution** → `/live` stays lightweight and can report `busy: true`; benchmarks run with timeout handling and the UI shows `Depth Engine: Busy` rather than immediately treating a benchmark as permanent backend loss.


### Native packaged resource verification and stale installs

The supported root native build scripts now fail unless both verification phases pass:

1. repo-root resources before electron-builder runs; and
2. packaged resource roots after electron-builder finishes.

Required native runtime resources are `backend/`, `backend/app.py`, `frontend/`, `frontend/index.html`, the platform virtualenv Python executable, `models/`, and `models/onnx/`. ONNX graph files such as `midas_small.onnx` remain optional unless a command explicitly uses `--onnx required` or `--onnx require-all`.

Direct Electron build commands are also guarded: `npm run build:mac:arm64`, `npm run build:win:arm64`, and `npm run build:linux:arm64` run the same repo-root and packaged-resource verification around the raw electron-builder command. The unsupported x64/universal scripts continue to fail with the unsupported-architecture helper.

If a packaged app under an installed location still reports missing resources after a successful rebuild, remove the stale installed copy and launch or reinstall the new artifact:

```bash
# macOS Apple Silicon
rm -rf "/Applications/DepthLens Pro.app"
scripts/build-native-macos.sh
open "electron-app/dist/mac-arm64/DepthLens Pro.app"
```

```powershell
# Windows ARM
# Uninstall the old DepthLens Pro from Settings, then rebuild and reinstall the new NSIS installer.
.\scripts\build-native-windows.ps1
```

```bash
# Linux ARM
scripts/build-native-linux.sh
./electron-app/dist/DepthLens\ Pro-*-linux-arm64.AppImage
```

### Backend Readiness and Stale Backend Recovery

**Symptom** → The app stays on “Starting engine” or reports no `/live` response. **Cause** → Port `8765` is occupied, the backend process exited before Uvicorn served `/live`, or the configured Python environment cannot import Uvicorn. **Resolution** → Run `cd electron-app && npm run backend:smoke`, then stop any non-DepthLens process that owns port `8765`.

**Symptom** → `/live` returns `ok`, but inference controls stay disabled. **Cause** → `/ready` found a missing required Python module such as `torch`, `cv2`, `PIL`, `fastapi`, `uvicorn`, or `numpy`. **Resolution** → Activate the repo-root virtual environment and run `python -m pip install -r backend/requirements.txt && venv/bin/python -m pip check`.

**Symptom** → Performance Analysis reports ONNX unavailable while PyTorch inference works. **Cause** → ONNX Runtime cannot import, cannot use the requested provider, or cannot load a valid graph from the configured path; benchmarks do not export missing graphs unless `DEPTHLENS_AUTO_EXPORT_ONNX=true` is set. **Resolution** → Run `venv/bin/python backend/scripts/export_onnx.py --all`, then inspect `curl http://127.0.0.1:8765/onnx/status?device=auto`.

**Symptom** → Redis errors appear in logs, but inference still works. **Cause** → Redis is optional and the cache service falls back to the in-memory cache after connection failures. **Resolution** → Use the app normally or configure Redis with `REDIS_URL` (`str | None`, default `None`) when a shared cache is required.

**Symptom** → GT upload fails for EXR, PFM, or a multi-channel NPY file. **Cause** → Ground truth decoding supports only `.png`, `.tif`/`.tiff`, and `.npy`, and NPY must be H×W or H×W×1. **Resolution** → Convert the label to PNG/TIFF grayscale or a numeric H×W/H×W×1 NPY file.

**Symptom** → A cached Redis value disappears and the request recomputes inference. **Cause** → The cache rejects legacy pickle payloads with `DLP1\0` or raw pickle magic and treats them as cache misses. **Resolution** → Clear old cache entries with `curl -X DELETE http://127.0.0.1:8765/cache`.

### Acceptance Checks

Run these commands while the app is open; `/live` should include `status`, `service`, `version`, `pid`, `timestamp`, and `uptime_seconds`, and `/devices` should include `devices`, `primary_device`, and `cached`.

```bash
curl --max-time 3 -v http://127.0.0.1:8765/live
curl --max-time 5 -v http://127.0.0.1:8765/devices
```

Run this command after quitting the app; the expected outcome is no process listening on port `8765`.

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

---

## Security

DepthLensPro keeps the desktop trust boundary narrow. The renderer cannot access Node.js directly and can only use explicitly exposed preload APIs.

- The renderer is sandboxed.
- `contextIsolation` is enabled.
- `nodeIntegration` is disabled.
- The preload bridge exposes only `getBackendUrl`, `getAppVersion`, `getPlatform`, `showSaveDialog`, `showOpenDialog`, `platform`, and `arch`.
- Navigation is allowed only to `http://127.0.0.1:<backendPort>` and the exact frontend `file://` path loaded by Electron.
- New windows are denied; `https:` and `mailto:` URLs open externally through Electron shell.
- The backend binds to `127.0.0.1` by default in local and desktop mode; Docker Compose overrides `HOST` to `0.0.0.0` only for container reachability.
- CORS keeps wildcard origins for the browser/file development workflow but does not allow credentialed CORS.
- Backend process cleanup only targets DepthLens-owned Uvicorn processes identified by metadata, command line, and project paths.
- The API is intended for local/trusted deployments; broader auth/admin gating and benchmark API redesign remain deferred rather than partially implemented.

To report a vulnerability, follow [`SECURITY.md`](SECURITY.md). Do not disclose security issues through public issue trackers.

---

## Contributing

1. Fork the repository.
2. Create a branch:

   ```bash
   git checkout -b feature/your-change
   ```

3. Make focused changes with tests or validation evidence.
4. Run the validation gates:

   ```bash
   black --check .
   ruff check .
   mypy backend/
   pytest
   ```

5. Commit, push, and open a pull request.

Preserve existing API response shapes and desktop workflow compatibility unless a breaking change has been explicitly reviewed.

---

## License

This project is licensed under the MIT License.

---

## Maintainer

**Ayushman Raha**<br>
GitHub: [https://github.com/AyushmanRaha](https://github.com/AyushmanRaha)

---

## Acknowledgements

- MiDaS depth estimation models by Intel ISL
- PyTorch, ONNX Runtime, OpenCV, FastAPI, Redis, Chart.js, Electron, and electron-builder communities
- PyTorch, ONNX Runtime, OpenCV, FastAPI, Redis, Chart.js, Electron, and electron-builder communities

## Depth engine readiness, ONNX assets, and PyTorch fallback

DepthLensPro can be live, inference-ready, and ONNX-ready as separate states. PyTorch remains the portable inference path, while ONNX Runtime acceleration depends on exported graph files and providers reported by the installed runtime.

### Model asset setup

Supported model assets use canonical lowercase IDs.

| Canonical ID | Display name | Torch Hub model | Default ONNX filename |
|---|---|---|---|
| `midas_small` | MiDaS Small | `MiDaS_small` | `midas_small.onnx` |
| `dpt_hybrid` | DPT Hybrid | `DPT_Hybrid` | `dpt_hybrid.onnx` |
| `dpt_large` | DPT Large | `DPT_Large` | `dpt_large.onnx` |

ONNX files resolve in this exact priority order:

1. `DEPTHLENSPRO_MODEL_DIR/onnx`, where `DEPTHLENSPRO_MODEL_DIR` is `str | None` with default `None`.
2. `DEPTHLENSPRO_MODEL_DIR`, using the same `str | None` value when set.
3. `DEPTHLENS_ONNX_DIR`, a `str | None` legacy override with default `None`.
4. `ONNX_WEIGHTS_DIR`, a `str | None` compatibility override with default `None`.
5. `models/onnx`, the repository default.
6. `backend/onnx_weights`, the legacy backend fallback.
7. `~/.cache/DepthLensPro/onnx`, the user cache fallback.

This command exports all supported ONNX graphs into the resolved default output directory and expects non-empty model files for `midas_small`, `dpt_hybrid`, and `dpt_large`.

```bash
venv/bin/python backend/scripts/export_onnx.py --all
```

This command checks ONNX runtime and path status for the automatic device selection.

```bash
curl http://127.0.0.1:8765/onnx/status?device=auto
```

If an ONNX file is missing, empty, invalid, or provider-incompatible, inference responses report metadata such as `engine_used`, `fallback_used`, `warnings`, `model_id`, `model_display_name`, and `device_used` instead of failing solely because ONNX is unavailable. Benchmark requests report missing ONNX weights unless `DEPTHLENS_AUTO_EXPORT_ONNX=true` explicitly opts into request-time export.

### Apple Silicon and provider notes

On macOS Apple Silicon, PyTorch can use MPS/Metal for the PyTorch path. ONNX Runtime uses only providers returned by `onnxruntime.get_available_providers()`, commonly `CPUExecutionProvider` unless the installed ONNX Runtime build includes another supported provider.

CUDA, XPU, CoreML, OpenVINO, ROCm, MIGraphX, DirectML, TensorRT, and CPU provider choices depend on the installed runtime packages and hardware. `/health` and `/onnx/status` report PyTorch devices separately from ONNX Runtime providers.

### Troubleshooting: `ONNXRuntimeError: model_path must not be empty`

**Symptom** → ONNX Runtime raises `model_path must not be empty`. **Cause** → The selected model did not resolve to an existing non-empty `.onnx` file in the ONNX search path. **Resolution** → Run `venv/bin/python backend/scripts/export_onnx.py --all` and then `curl http://127.0.0.1:8765/onnx/status?device=auto`.

### Troubleshooting: ONNX export failed with SymInt / `torch.export`

**Symptom** → ONNX export fails with `SymInt`, reshape, unflatten, or `torch.export` errors. **Cause** → Some MiDaS/DPT graphs do not export cleanly through every PyTorch ONNX path. **Resolution** → Keep PyTorch inference enabled, inspect the export error, and retry `venv/bin/python backend/scripts/export_onnx.py --all --force` after updating compatible ONNX tooling.

### Verification commands

This command verifies the Python package environment without running tests or downloads beyond installed dependencies.

```bash
venv/bin/python -m pip check
```

This command verifies Electron resource paths before packaging.

```bash
cd electron-app
npm run verify:resources
```

## Cross-platform ARM native and terminal runbook

DepthLens Pro native packaging is intentionally limited to ARM targets: macOS Apple Silicon (`darwin arm64`), Windows ARM (`win32 arm64`), and Linux ARM (`linux arm64`/`aarch64`). Intel Mac, Windows x64, Linux x64, and macOS universal packages are unsupported and must remain blocked by setup, build, and Electron startup checks. ONNX options are identical on all setup and native build entrypoints: `--with-onnx`, `--without-onnx`, `--onnx-models midas_small`, `--onnx-models dpt_hybrid`, `--onnx-models dpt_large`, `--onnx-models midas_small,dpt_hybrid`, `--onnx-models all`, `--onnx-strict`, `--onnx-validate-only`, and `--onnx-force`.

### Native app clone-to-run recipes

#### A. macOS Apple Silicon native app without ONNX
```bash
python3 --version
node --version
npm --version
cd ~/Developer
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
scripts/build-native-macos.sh --without-onnx
cd electron-app && npm run verify:resources:native && node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --onnx off
open "dist/mac-arm64/DepthLens Pro.app"
```
Artifact: `electron-app/dist/mac-arm64/DepthLens Pro.app` and a DMG in `electron-app/dist/`. To replace a stale install, quit the app, remove `/Applications/DepthLens Pro.app`, and copy/open the fresh artifact.

#### B. macOS Apple Silicon native app with MiDaS Small ONNX
```bash
cd ~/Developer
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
scripts/build-native-macos.sh --with-onnx --onnx-models midas_small --onnx-strict
cd electron-app && node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --onnx required --models midas_small
open "dist/mac-arm64/DepthLens Pro.app"
```

#### C. macOS Apple Silicon native app with all ONNX weights
```bash
cd ~/Developer
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
scripts/build-native-macos.sh --with-onnx --onnx-models all --onnx-strict
cd electron-app && node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --onnx require-all --models all
open "dist/mac-arm64/DepthLens Pro.app"
```

#### D. Windows ARM native app without ONNX
```powershell
python --version
node --version
npm --version
cd $HOME\source
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
.\scripts\build-native-windows.ps1 --without-onnx
cd electron-app
node scripts/verify-packaged-resources.js --platform win32 --arch arm64 --mode native --onnx off
& ".\dist\win-arm64-unpacked\DepthLens Pro.exe"
```
Artifact: `electron-app\dist\win-arm64-unpacked` and an NSIS installer in `electron-app\dist`. Replace stale installs from Settings > Apps or remove `%LOCALAPPDATA%\Programs\DepthLens Pro` before installing/opening the fresh build.

#### E. Windows ARM native app with MiDaS Small ONNX
```powershell
cd $HOME\source
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
.\scripts\build-native-windows.ps1 --with-onnx --onnx-models midas_small --onnx-strict
cd electron-app
node scripts/verify-packaged-resources.js --platform win32 --arch arm64 --mode native --onnx required --models midas_small
& ".\dist\win-arm64-unpacked\DepthLens Pro.exe"
```

#### F. Windows ARM native app with all ONNX weights
```powershell
cd $HOME\source
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
.\scripts\build-native-windows.ps1 --with-onnx --onnx-models all --onnx-strict
cd electron-app
node scripts/verify-packaged-resources.js --platform win32 --arch arm64 --mode native --onnx require-all --models all
& ".\dist\win-arm64-unpacked\DepthLens Pro.exe"
```

#### G. Linux ARM native app without ONNX
```bash
python3 --version
node --version
npm --version
cd ~/src
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
scripts/build-native-linux.sh --without-onnx
cd electron-app && node scripts/verify-packaged-resources.js --platform linux --arch arm64 --mode native --onnx off
chmod +x dist/*arm64*.AppImage && ./dist/*arm64*.AppImage
```
Artifact: `electron-app/dist/*arm64*.AppImage` and unpacked resources when retained by electron-builder. Replace stale desktop entries/AppImages under `/opt`, `/usr/local/bin`, or `~/.local/share/applications` before launching the fresh build.

#### H. Linux ARM native app with MiDaS Small ONNX
```bash
cd ~/src
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
scripts/build-native-linux.sh --with-onnx --onnx-models midas_small --onnx-strict
cd electron-app && node scripts/verify-packaged-resources.js --platform linux --arch arm64 --mode native --onnx required --models midas_small
chmod +x dist/*arm64*.AppImage && ./dist/*arm64*.AppImage
```

#### I. Linux ARM native app with all ONNX weights
```bash
cd ~/src
git clone https://github.com/<owner>/DepthLensPro.git
cd DepthLensPro
scripts/build-native-linux.sh --with-onnx --onnx-models all --onnx-strict
cd electron-app && node scripts/verify-packaged-resources.js --platform linux --arch arm64 --mode native --onnx require-all --models all
chmod +x dist/*arm64*.AppImage && ./dist/*arm64*.AppImage
```

### Terminal-only backend/frontend runbook

Do not start a second backend on port `8765`; run `python scripts/diagnose_backend.py` first if unsure.

#### J. macOS terminal-only without ONNX
```bash
cd ~/Developer && git clone https://github.com/<owner>/DepthLensPro.git && cd DepthLensPro
scripts/setup-macos.sh --without-onnx
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
Terminal 2:
```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/onnx/status
cd electron-app && npm run start:dev
```
Stop with `Ctrl-C` in Terminal 1.

#### K. macOS terminal-only with ONNX
```bash
cd ~/Developer && git clone https://github.com/<owner>/DepthLensPro.git && cd DepthLensPro
scripts/setup-macos.sh --with-onnx --onnx-models midas_small --onnx-strict
venv/bin/python backend/scripts/export_onnx.py --validate-only --model midas_small --strict
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
Use the same Terminal 2 `curl` and `npm run start:dev` commands as above.

#### L. Windows ARM terminal-only without ONNX
```powershell
cd $HOME\source; git clone https://github.com/<owner>/DepthLensPro.git; cd DepthLensPro
.\scripts\setup-windows.ps1 --without-onnx
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
Terminal 2:
```powershell
Invoke-RestMethod http://127.0.0.1:8765/live
Invoke-RestMethod http://127.0.0.1:8765/ready
Invoke-RestMethod http://127.0.0.1:8765/onnx/status
cd electron-app; npm run start:dev
```
Stop with `Ctrl-C` in Terminal 1.

#### M. Windows ARM terminal-only with ONNX
```powershell
cd $HOME\source; git clone https://github.com/<owner>/DepthLensPro.git; cd DepthLensPro
.\scripts\setup-windows.ps1 --with-onnx --onnx-models midas_small --onnx-strict
.\venv\Scripts\python.exe backend\scripts\export_onnx.py --validate-only --model midas_small --strict
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
Use the same Terminal 2 `Invoke-RestMethod` and `npm run start:dev` commands as above.

#### N. Linux ARM terminal-only without ONNX
```bash
cd ~/src && git clone https://github.com/<owner>/DepthLensPro.git && cd DepthLensPro
scripts/setup-linux.sh --without-onnx
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
Terminal 2:
```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/onnx/status
cd electron-app && npm run start:dev
```
Stop with `Ctrl-C` in Terminal 1.

#### O. Linux ARM terminal-only with ONNX
```bash
cd ~/src && git clone https://github.com/<owner>/DepthLensPro.git && cd DepthLensPro
scripts/setup-linux.sh --with-onnx --onnx-models midas_small --onnx-strict
venv/bin/python backend/scripts/export_onnx.py --validate-only --model midas_small --strict
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
Use the same Terminal 2 `curl` and `npm run start:dev` commands as above.

### Cross-platform diagnostics and troubleshooting

Run diagnostics from the repo root on every supported platform:
```bash
python scripts/diagnose_backend.py
```
Windows equivalent:
```powershell
python .\scripts\diagnose_backend.py
```
The report includes platform, architecture, repo root, Python and venv Python versions, Node/npm versions, backend port occupancy, PID/command line where available, `/live`, `/ready`, `/onnx/status`, model and ONNX directories, ONNX file status for `midas_small`, `dpt_hybrid`, `dpt_large`, Electron log path, and platform-specific remediation commands. If PID discovery is unavailable, the diagnostic continues with a warning.

Troubleshooting command blocks:

- **Depth Engine Offline or `/live` timeout**: run `python scripts/diagnose_backend.py`, verify `/live`, and start only one backend on port `8765`.
- **Port `8765` already in use**: macOS/Linux use `lsof -nP -iTCP:8765 -sTCP:LISTEN` when available or the diagnostic fallback; Windows use `Get-NetTCPConnection -LocalPort 8765 -State Listen` or the diagnostic fallback. Kill only a confirmed stale DepthLens PID.
- **Stale backend PID files**: remove `backend.pid` and `backend.json` under the Electron user-data directory reported in diagnostics, then restart.
- **Stale installed app copy**: replace `/Applications/DepthLens Pro.app` on macOS, uninstall/remove `%LOCALAPPDATA%\Programs\DepthLens Pro` on Windows, or replace old AppImage/desktop entries on Linux.
- **ONNX files present in repo but missing in packaged app**: rebuild with `--with-onnx --onnx-models <selection>` and re-run `node scripts/verify-packaged-resources.js --platform <darwin|win32|linux> --arch arm64 --mode native --onnx required --models <selection>`.
- **Redis unavailable**: use `DEPTHLENS_CACHE_BACKEND=memory`; Redis is optional for local setup.
- **ONNX/CoreML warnings on macOS**: PyTorch MPS and ONNX Runtime providers are separate; `/onnx/status` reports the available provider fallback.
- **ONNX provider fallback on Windows/Linux**: CPU fallback is expected unless the installed ONNX Runtime exposes DirectML, CUDA, OpenVINO, or another provider.
- **DPT Hybrid/Large export warnings**: these large exports are optional unless `--onnx-strict`/`--onnx-models all` is used; PyTorch fallback remains available.
- **Python dependency/import failure**: rerun the platform setup script and `venv/bin/python -m pip check` or `.\venv\Scripts\python.exe -m pip check`.
- **npm install failure**: verify Node/npm, remove stale `electron-app/node_modules`, and rerun the setup script.
- **zsh comment paste issue on macOS**: run `setopt interactivecomments` before pasting commented shell blocks.
- **PowerShell execution policy on Windows**: run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current shell.
- **Linux missing `python3.12-venv`/distro package**: install the distro's matching Python venv package, then rerun `scripts/setup-linux.sh`.

### Cross-platform PR acceptance criteria

- macOS ARM native build still works.
- Windows ARM native build still works.
- Linux ARM native build still works.
- ONNX prompt/flags behave consistently across macOS, Windows, and Linux.
- Packaged ONNX verification works across macOS, Windows, and Linux.
- `/live` is lightweight and platform-independent.
- Electron startup recovery works across macOS, Windows, and Linux.
- README contains complete clone-to-run command sequences for every supported platform and workflow.
- No new macOS-only assumptions are introduced into shared setup, backend, frontend, Electron lifecycle, diagnostics, or verification code.
