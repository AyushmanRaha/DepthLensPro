# DepthLensPro

[![CI/CD Status](https://img.shields.io/badge/CI%2FCD-passing-placeholder.svg)](#validation--ci)
[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](electron-app/package.json)
[![API](https://img.shields.io/badge/API-3.1.0-6f42c1.svg)](backend/main.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64%20%7C%20Windows-lightgrey.svg)](#end-user-installation)

Last Updated: 10th July 2026

DepthLensPro is a native cross-platform desktop application for hardware-accelerated monocular depth estimation from 2D images. It pairs an Electron desktop client with a FastAPI inference service that runs MiDaS-family models through PyTorch and ONNX Runtime, with Redis-backed caching for repeatable low-latency workflows.

The application is optimized for local-first image processing, secure desktop packaging, production-oriented diagnostics, and explicit hardware routing across Apple Silicon, CUDA-capable GPUs, Intel XPU, and CPU fallback paths.

---

## Table of Contents

- [Capability Overview](#capability-overview)
- [Hardware Acceleration](#hardware-acceleration)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Observability & Telemetry](#observability--telemetry)
- [Quick Start / Runbook](#quick-start--runbook)
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

- Monocular depth estimation using MiDaS-family models: `MiDaS_small`, `DPT_Hybrid`, and `DPT_Large`.
- ONNX Runtime execution when exported static graphs are available, with automatic PyTorch fallback.
- Single-image and batch inference, with batch requests capped at **10 images**.
- Supported uploads: `PNG`, `JPG/JPEG`, `WEBP`, and `BMP`.
- Dual output generation: colorized depth map and grayscale depth map.
- Colormap options: `inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, and `turbo`.
- Runtime model comparison across supported architectures with side-by-side outputs and metric charts.
- Depth-quality and consistency metrics including SSIM, SILog, PSNR, entropy, dynamic range, histogram coverage, gradient statistics, and edge density.

### System Architecture & Observability

- FastAPI ASGI backend with explicit route, inference, cache, benchmark, configuration, and hardware modules.
- Redis-backed inference cache keyed by image, model, colormap, and device, with TTL control and in-memory fallback.
- Structured runtime configuration through environment variables and optional `.env` values.
- JSON structured logging for collector-friendly backend logs.
- Health, benchmark, cache, memory, disk, acceleration, and device diagnostics exposed through API endpoints.
- Docker and Docker Compose support for backend-plus-Redis execution.

### Desktop Client

- Electron desktop shell with automatic FastAPI child-process lifecycle management.
- Sandboxed renderer, `contextIsolation` enabled, `nodeIntegration` disabled, and navigation constrained to loopback origins.
- Splash screen during backend warm-up, followed by a workspace once the engine is ready or timeout handling completes.
- Drag-and-drop upload workflow with progress, adaptive ETA, cancellation, result cards, lightbox metrics, and download actions.
- Persistent user preferences for model, colormap, theme, and compute device.
- Responsive workspace panels for generation, model comparison, session analytics, and application details.
- Animated landing experience with theme-aware vector background and liquid-metal logo transition.

---

## Hardware Acceleration

DepthLensPro treats accelerator selection as an explicit runtime concern. Users can select `Auto`, `CPU`, `CUDA`, `MPS`, or `XPU`, while the backend validates availability and falls back safely when a provider is unavailable.

| Target | Runtime Path | Notes |
|---|---|---|
| **Apple Silicon** | PyTorch MPS via Metal | Native `arm64` packaging for macOS with zero-translation execution on Apple Silicon. |
| **NVIDIA GPU** | PyTorch CUDA and ONNX Runtime provider support where installed | Best suited for larger DPT models and high-throughput workloads. |
| **Intel XPU** | PyTorch XPU where available | Exposed as a first-class selectable compute target when detected. |
| **CPU** | PyTorch CPU and ONNX Runtime CPU provider | Stable fallback path for all supported systems. |
| **ONNX Runtime** | Static `.onnx` graphs with provider fallback | Use `backend/scripts/export_onnx.py` to export model weights for accelerated inference paths. |

The macOS desktop build targets **macOS arm64** and runs PyTorch MPS inference through Metal without Rosetta translation overhead. ONNX Runtime remains available as an alternative execution path when static model graphs and compatible providers are present.

---

## Architecture

DepthLensPro separates user interface, privilege boundaries, and inference workloads into independently reasoned components:

```text
Electron Main Process
  ├─ owns native window lifecycle
  ├─ starts/stops the FastAPI backend as a child process
  └─ enforces navigation and packaging behavior

Electron Sandboxed Renderer
  ├─ renders upload, comparison, metrics, and analytics workflows
  ├─ communicates only through approved browser APIs and preload bridge
  └─ persists non-sensitive preferences in localStorage

Preload Bridge
  ├─ exposes a minimal contextBridge API
  └─ provides platform/architecture metadata under context isolation

FastAPI ASGI Backend
  ├─ serves inference, health, devices, models, benchmarks, and cache endpoints
  ├─ loads MiDaS models through PyTorch or exported ONNX graphs
  ├─ routes requests to CPU, MPS, CUDA, or XPU devices
  └─ integrates Redis cache, structured logging, and telemetry
```

### Separation of Concerns

- **Electron renderer** owns presentation, user interaction, progress state, charts, and local preferences.
- **Preload script** preserves context isolation by exposing only the minimal desktop metadata required by the UI.
- **Electron main process** owns privileged desktop capabilities, backend process management, packaged-resource resolution, and navigation policy.
- **FastAPI backend** owns model lifecycle, inference execution, metrics generation, diagnostics, and cache coordination.

---

## Technology Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Desktop shell | Electron `42.x`, electron-builder `26.x` | Native packaging, window lifecycle, backend child process orchestration |
| Renderer | HTML5, CSS3, JavaScript ES2022, Chart.js | Workspace UI, comparison views, analytics, result rendering |
| API service | Python `3.10+`, FastAPI, Uvicorn | ASGI API, request validation, lifecycle hooks, diagnostics |
| AI runtime | PyTorch, Torch Hub, ONNX, ONNX Runtime | MiDaS model loading and accelerated inference |
| Image processing | OpenCV, NumPy, Pillow | Input normalization, depth-map post-processing, PNG encoding |
| Cache | Redis with in-memory fallback | TTL-based reuse of repeated inference results |
| Configuration | `pydantic-settings` with fallback shim | Environment and `.env` driven backend configuration |
| Observability | JSON logging, `/health`, `/benchmark`, `/cache/metrics` | Runtime state, resource telemetry, benchmark reporting |
| Delivery | Docker, Docker Compose, GitHub Actions | Containerized backend, Redis stack, quality gates |

---

## Observability & Telemetry

DepthLensPro exposes operational signals suitable for local diagnostics, release validation, and production-style monitoring.

| Signal | Surface | Description |
|---|---|---|
| Structured logs | Backend stdout | JSON records with timestamp, level, logger, module, function, line, process, thread, and exception details. |
| Health state | `GET /health` | Engine status, primary device, loaded models, PyTorch version, acceleration checks, device inventory, cache summary, and system metadata. |
| Memory telemetry | `GET /health` | Memory status, pressure percentage, limit threshold, total bytes, available bytes, and used bytes. |
| Disk telemetry | `GET /health` | Disk status, monitored path, usage percentage, limit threshold, total bytes, free bytes, and used bytes. |
| Cache metrics | `GET /cache/metrics` and `GET /health` | Redis availability, backend type, hit/miss counts, fallback failures, keyspace size, and TTL. |
| Runtime benchmarks | `GET /benchmark` and `GET /api/benchmark` | PyTorch versus ONNX Runtime latency, throughput, memory, and speedup comparison. |
| Session analytics | Desktop workspace | Processed image count, latency history, average/min/max latency, throughput, total inference time, cache hits, and error count. |

---

## Quick Start / Runbook

DepthLensPro uses a local FastAPI backend at `http://127.0.0.1:8765` by default. The Electron app starts that backend automatically in development and packaged desktop modes, then exposes the resolved URL to the renderer through the secure preload bridge.

> **First run:** model weights may be downloaded through `torch.hub` the first time a model is loaded. Subsequent runs use the local model cache.
>
> **Cache note:** Redis is optional. If Redis is unavailable or not installed, the backend logs the condition and falls back to an in-memory cache automatically.

### Prerequisites

- Python `3.10+`
- `pip`
- Node.js and npm
- Git
- Optional: Redis for distributed cache validation

### A. Native App Build / Install by Platform

#### macOS Native App Build

Builds an unsigned Apple Silicon DMG. For reproducible packaged builds, create the repo-root `venv/` before running electron-builder because `electron-app/package.json` includes `../venv` as an `extraResources` entry.

```bash
cd DepthLensPro

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

cd electron-app
npm install
npm run build:mac
```

Open the generated DMG from `electron-app/dist/`, drag **DepthLens Pro** to `/Applications`, and launch it. If macOS Gatekeeper blocks an unsigned local build, clear quarantine metadata:

```bash
xattr -cr "/Applications/DepthLens Pro.app"
```

#### Windows Native App Build

Builds an NSIS installer. Run these commands from PowerShell so the Windows virtual environment layout is used (`venv\Scripts\python.exe`).

```powershell
cd DepthLensPro

py -3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

cd electron-app
npm install
npm run build:win
```

Run the generated installer from `electron-app\dist\`, then launch **DepthLens Pro** from the Start menu or desktop shortcut. Packaged builds expect the prepared `venv/`, `backend/`, and `frontend/` resources to be present in the installer resources.

#### End-User Installation from Release Artifacts

- **macOS:** Download the latest `DMG`, mount it, drag **DepthLens Pro** into `/Applications`, and launch the app.
- **Windows:** Download the latest Windows installer (`.exe`), run it, and launch **DepthLens Pro** from the Start menu or desktop shortcut.

### B. Terminal-Only Local Test / Development

Use this path when you want to test the app locally without creating a native installer. `npm start` launches Electron, starts FastAPI on `127.0.0.1:8765`, and keeps the backend URL synchronized with the frontend.

#### macOS / Linux Terminal-Only Development Test

```bash
cd DepthLensPro

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

cd electron-app
npm install
npm start
```

#### Windows PowerShell Terminal-Only Development Test

```powershell
cd DepthLensPro

py -3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

cd electron-app
npm install
npm start
```

### Backend-Only Smoke Test

Run the backend without Electron when validating API routes or working in a terminal-only environment.

#### macOS / Linux

```bash
cd DepthLensPro
source venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

#### Windows PowerShell

```powershell
cd DepthLensPro
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

Health and route checks:

```bash
curl http://127.0.0.1:8765/
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/devices
```

From `electron-app/`, you can also run the npm smoke helper while the backend is running:

```bash
npm run backend:smoke
```

### Docker Compose

```bash
docker compose up --build
```

---

## API Surface

Base URL: `http://127.0.0.1:8765`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API metadata and service version. |
| `GET` | `/health` | Engine status, devices, acceleration checks, telemetry, cache metrics, and runtime metadata. |
| `GET` | `/devices` | Discovered compute targets with type, compute class, and hardware details. |
| `GET` | `/models` | Supported model registry with runtime notes. |
| `GET` | `/colormaps` | Supported colormap keys. |
| `POST` | `/estimate` | Single-image depth estimation. Form fields: `file`, `model`, `colormap`, `device`. |
| `POST` | `/batch` | Multi-image depth estimation for up to 10 files. Uses the same form fields as `/estimate`. |
| `GET` | `/benchmark` | PyTorch versus ONNX Runtime benchmark summary. |
| `GET` | `/api/benchmark` | Compatibility alias for benchmark summary. |
| `GET` | `/cache/metrics` | Live Redis and fallback cache metrics. |
| `DELETE` | `/cache` | Clear Redis-backed and in-memory inference cache entries. |

<details>
<summary><strong>Example <code>/health</code> response</strong></summary>

```json
{
  "status": "ok",
  "version": "3.1.0",
  "primary_device": "mps",
  "devices": { "mps": { "available": true, "type": "GPU" } },
  "loaded_models": ["MiDaS_small:mps"],
  "cache_entries": 3,
  "cache_metrics": {
    "backend": "redis",
    "redis_available": true,
    "total_hits": 12,
    "cache_misses": 4,
    "keyspace_size": 3,
    "ttl_seconds": 3600
  },
  "torch_version": "2.11.0",
  "cuda_available": false,
  "mps_available": true,
  "xpu_available": false,
  "acceleration_ok": true,
  "acceleration_checks": {
    "cuda": { "available": false, "operational": false },
    "mps": { "available": true, "operational": true },
    "xpu": { "available": false, "operational": false }
  },
  "telemetry": {
    "memory": {
      "status": "ok",
      "pressure_percent": 42.1,
      "limit_percent": 90.0,
      "total_bytes": 17179869184,
      "available_bytes": 9932800000,
      "used_bytes": 7247069184
    },
    "disk": {
      "status": "ok",
      "path": "/",
      "usage_percent": 61.3,
      "limit_percent": 90.0,
      "total_bytes": 499963174912,
      "free_bytes": 193481154560,
      "used_bytes": 306482020352
    }
  },
  "system": {
    "os": "macOS-15.x-arm64",
    "machine": "arm64",
    "cpu": "Apple M3 Pro",
    "accelerators": ["GPU · Apple M3 Pro (Metal)"]
  }
}
```

</details>

<details>
<summary><strong>Example <code>/benchmark</code> response</strong></summary>

```json
{
  "model": "MiDaS_small",
  "device_requested": "auto",
  "device_resolved": "mps",
  "iterations": 3,
  "frame_shape": [384, 384, 3],
  "weights": {
    "onnx_path": "backend/onnx_weights/MiDaS_small.onnx",
    "onnx_available": true
  },
  "results": [
    {
      "engine": "pytorch",
      "status": "ok",
      "latency_ms": { "avg": 31.4, "min": 30.8, "max": 32.2 },
      "throughput_fps": 31.85
    },
    {
      "engine": "onnxruntime",
      "status": "ok",
      "latency_ms": { "avg": 18.6, "min": 18.1, "max": 19.0 },
      "throughput_fps": 53.76
    }
  ],
  "comparison": {
    "latency_delta_ms": 12.8,
    "speedup": 1.69,
    "faster_engine": "onnxruntime"
  }
}
```

</details>

<details>
<summary><strong>Example <code>/estimate</code> response</strong></summary>

```json
{
  "depth_map": "<base64 PNG>",
  "grayscale": "<base64 PNG>",
  "metrics": {
    "ssim": 0.812,
    "silog": 7.43,
    "psnr": 28.5,
    "entropy": 6.82,
    "dynamic_range": 8.14
  },
  "latency_ms": 31.4,
  "model": "MiDaS_small",
  "colormap": "inferno",
  "device_used": "mps",
  "resolution": { "width": 1024, "height": 768 },
  "filename": "photo.jpg",
  "cached": false
}
```

</details>

---

## Metrics & Models

### Metrics Reference

| Metric | Category | Description |
|---|---|---|
| `ssim` | No ground truth required | Structural similarity between grayscale input and predicted depth map. |
| `silog` | No ground truth required | Scale-invariant log-depth variance after global scale normalization. |
| `psnr` | No ground truth required | Peak signal-to-noise ratio of the depth map relative to its mean. |
| `entropy` | No ground truth required | Shannon entropy of the predicted depth histogram. |
| `dynamic_range` | No ground truth required | Log-ratio of maximum to minimum non-zero depth. |
| `coverage` | No ground truth required | Fraction of histogram bins above the coverage threshold. |
| `gradient_mean` / `gradient_std` | No ground truth required | Sobel-gradient strength and variation in the generated depth map. |
| `edge_density` | No ground truth required | Fraction of pixels with gradient magnitude above adaptive threshold. |
| `mae` / `rmse` / `log_rmse` | Self-consistency proxy | Error metrics relative to the predicted depth mean. |
| `abs_rel`, `sq_rel`, `delta_1/2/3`, `lpips`, `ordinal_error`, `surface_normal_error` | Ground truth required | Reported as unavailable unless a ground-truth depth reference is supplied by a future workflow. |

### Model Reference

| Model | Architecture | Approximate GPU Latency | Quality Profile | Recommended Compute |
|---|---|---:|---|---|
| `MiDaS_small` | EfficientNet-Lite | ~30 ms | Good | CPU or GPU |
| `DPT_Hybrid` | ViT-Hybrid | ~120 ms | Very good | GPU recommended |
| `DPT_Large` | ViT-Large | ~400 ms | Highest | GPU required for practical interactivity |

---

## Project Structure

```text
DepthLensPro/
├── backend/                         # FastAPI inference engine
│   ├── app.py                       # Backward-compatible ASGI entrypoint
│   ├── main.py                      # App factory, CORS, JSON logging, lifespan hooks
│   ├── config.py                    # Environment-backed settings
│   ├── depth_models.py              # Legacy estimator and ONNX Runtime engine
│   ├── api/routes.py                # HTTP route handlers
│   ├── services/
│   │   ├── benchmarks.py            # Runtime benchmark service
│   │   ├── cache_service.py         # Redis cache and memory fallback
│   │   └── inference.py             # Model loading, image pipeline, metrics
│   ├── utils/hardware.py            # Device discovery and acceleration checks
│   ├── scripts/export_onnx.py       # MiDaS-to-ONNX export utility
│   └── tests/                       # Backend test suite
├── frontend/                        # Renderer UI assets
│   ├── index.html                   # Workspace and panel structure
│   ├── style.css                    # Theme and layout system
│   ├── script.js                    # UI state, API calls, charts, metrics
│   └── welcome-anim.js              # Landing animation canvas
├── electron-app/                    # Electron desktop shell
│   ├── main.js                      # Main process and backend lifecycle
│   ├── preload.js                   # Context-isolated bridge
│   ├── package.json                 # Scripts and build configuration
│   └── src/splash.html              # Backend startup splash screen
├── .github/workflows/ci.yml         # Quality gates
├── Dockerfile                       # Backend container image
├── docker-compose.yml               # Backend and Redis stack
├── pyproject.toml                   # Black, Ruff, and pytest configuration
├── mypy.ini                         # Static typing configuration
├── CONTRIBUTING.md                  # Contribution process
├── SECURITY.md                      # Vulnerability reporting
└── README.md
```

---

## Validation & CI

Run the local quality gates before opening a pull request:

```bash
black --check .
ruff check .
mypy backend/
pytest
```

The GitHub Actions pipeline runs the same core checks for formatting, linting, typing, and tests. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution requirements.

---

## Troubleshooting

| Issue | Recommended Action |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and rerun `pip install -r backend/requirements.txt`. |
| Frontend reports that the engine is offline | Confirm the backend is listening on `127.0.0.1:8765` and check `curl http://127.0.0.1:8765/health`. |
| Browser CORS or local file issue | Serve the frontend through a local HTTP server instead of opening files directly through `file://`. |
| Slow inference | Use `MiDaS_small`, select an available accelerator, or export ONNX weights for the target model. |
| Port already in use | Start Uvicorn with a different `--port` value or free port `8765`. |
| First run is slow | Allow the initial model-weight download to complete; later runs use the local Torch cache. |
| macOS blocks launch | Run `xattr -cr "/Applications/DepthLens Pro.app"` for unsigned local builds. |
| Packaged backend does not start | Check the desktop logs at `~/Library/Logs/depthlens-pro/main.log`. |
| Python not found in packaged app | Verify the `venv/` folder is included in electron-builder `extraResources`. |
| Duplicate app icon on macOS | Remove the old application bundle and reinstall from the latest DMG. |
| `acceleration_ok: false` | A GPU backend failed the runtime probe; CPU inference remains available. |
| `/health` reports degraded status | Inspect memory pressure and disk usage telemetry for threshold violations. |
| Redis cache unavailable | Verify Redis host, port, credentials, and container health; the backend uses in-memory fallback automatically. |
| `/benchmark` reports ONNX unavailable | Export the required graph or set `DEPTHLENS_ONNX_DIR` to the directory containing `.onnx` files. |

---

## Security

DepthLensPro is designed as a local-first desktop application with a constrained trust boundary:

- The renderer is sandboxed.
- `contextIsolation` is enabled.
- `nodeIntegration` is disabled.
- Navigation is restricted to `127.0.0.1` and `localhost`.
- The backend binds to loopback in desktop mode.
- The preload bridge exposes only minimal platform and architecture metadata.

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
