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

DepthLensPro uses a local FastAPI backend at `http://127.0.0.1:8765` by default. Electron starts that backend automatically, waits for `/live`, and passes the resolved backend URL to the renderer through the preload bridge.

The first renderer pass calls `/ready` before enabling inference controls, then loads `/health`, `/devices`, and `/cache/metrics` as background diagnostics. Redis is optional; when Redis is unavailable, the cache service uses its in-memory backend.

### Prerequisites

- Python `3.10` through `3.12`
- `pip`
- Node.js and npm
- Git

### A. Get the Repository in Downloads

Keep the working copy under the user Downloads folder on every supported platform. The repository URL is a placeholder because this checkout does not declare a public remote.

#### macOS / Linux

This command clones the repository into `~/Downloads/DepthLensPro` and changes into the project root.

```bash
cd ~/Downloads
git clone https://github.com/<your-org-or-user>/DepthLensPro.git
cd DepthLensPro
```

#### Windows PowerShell

This command clones the repository into `%USERPROFILE%\Downloads\DepthLensPro` and changes into the project root.

```powershell
cd "$env:USERPROFILE\Downloads"
git clone https://github.com/<your-org-or-user>/DepthLensPro.git
cd DepthLensPro
```

#### Windows Command Prompt

This command clones the repository into `%USERPROFILE%\Downloads\DepthLensPro` and changes into the project root.

```bat
cd %USERPROFILE%\Downloads
git clone https://github.com/<your-org-or-user>/DepthLensPro.git
cd DepthLensPro
```

### B. Install and Verify Backend Dependencies

Run this section after a fresh clone, branch switch, or dependency update. The commands create the repo-root Python virtual environment, install backend Python packages, verify dependency consistency, install Electron packages, and verify packaged resource paths.

#### macOS

This command prepares Python and Node dependencies on macOS, and the expected outcome is a valid virtual environment plus verified Electron resources.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
```

#### Linux ARM

This command prepares Python and Node dependencies on Linux ARM, and the expected outcome is a valid virtual environment plus verified Electron resources.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
```

#### Windows PowerShell

This command prepares Python and Node dependencies, and the expected outcome is a valid virtual environment plus verified Electron resources.

```powershell
if (!(Test-Path venv)) { py -m venv venv }
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
```

#### Windows Command Prompt

This command prepares Python and Node dependencies, and the expected outcome is a valid virtual environment plus verified Electron resources.

```bat
if not exist venv py -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
```

### C. Native App Build / Install by Platform

Native builds package the backend, frontend, and repo-root virtual environment into an ARM desktop artifact. Each flow exports all ONNX graphs after resource verification so Performance Analysis works in the packaged app without a benchmark-triggered export.

#### macOS Native App Build

This command installs dependencies, exports all ONNX graphs, and builds the Apple Silicon DMG and app bundle.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend/scripts/export_onnx.py --all
cd electron-app
npm run build:mac:arm64
```

#### Windows Native App Build

This PowerShell command installs dependencies, exports all ONNX graphs, and builds the Windows ARM NSIS installer.

```powershell
if (!(Test-Path venv)) { py -m venv venv }
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend/scripts/export_onnx.py --all
cd electron-app
npm run build:win:arm64
```

This Command Prompt command installs dependencies, exports all ONNX graphs, and builds the Windows ARM NSIS installer.

```bat
if not exist venv py -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend\scripts\export_onnx.py --all
cd electron-app
npm run build:win:arm64
```

#### Linux ARM Native App Build

This command installs dependencies, exports all ONNX graphs, and builds the Linux ARM AppImage.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend/scripts/export_onnx.py --all
cd electron-app
npm run build:linux:arm64
```

#### Native App Development Test

This command starts the Electron development app after installing and verifying dependencies.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
npm start
```

#### End-User Installation from Release Artifacts

Install the ARM artifact for your platform and launch `DepthLens Pro`. The app starts FastAPI on loopback, waits for `/live`, then enables renderer controls only after `/ready` reports required inference dependencies.

### D. Terminal-Only Local Test / Development

Terminal-only mode starts the Electron app from source. Each platform block installs Python and Node dependencies, verifies the Python environment, exports ONNX graphs, installs Electron dependencies, and starts the app.

#### macOS

This command prepares all dependencies on macOS, exports ONNX graphs, and starts Electron from source.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
python backend/scripts/export_onnx.py --all
cd electron-app
npm install
npm run verify:resources
npm start
```

#### Linux ARM

This command prepares all dependencies on Linux ARM, exports ONNX graphs, and starts Electron from source.

```bash
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
python backend/scripts/export_onnx.py --all
cd electron-app
npm install
npm run verify:resources
npm start
```

#### Windows PowerShell

This command prepares all dependencies, exports ONNX graphs, and starts Electron from source.

```powershell
if (!(Test-Path venv)) { py -m venv venv }
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
python backend/scripts/export_onnx.py --all
cd electron-app
npm install
npm run verify:resources
npm start
```

#### Windows Command Prompt

This command prepares all dependencies, exports ONNX graphs, and starts Electron from source.

```bat
if not exist venv py -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip check
python backend\scripts\export_onnx.py --all
cd electron-app
npm install
npm run verify:resources
npm start
```

#### Backend and Static Frontend in Separate Windows

Use this mode only when debugging backend or renderer code without Electron lifecycle management. The backend serves API traffic on loopback, and the static frontend can be opened from disk or hosted separately for renderer-only inspection.

This command starts the backend directly with Uvicorn and expects `/live` to answer on port `8765`.

```bash
source venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8765 --workers 1
```

This command starts a simple static frontend server and expects the UI files to be available on port `5173`.

```bash
cd frontend
python -m http.server 5173
```

### E. Update, Rebuild, and Run the Latest Code

Use these flows when you already have a checkout and want a clean rebuild. They preserve the same dependency order as the first-run blocks.

#### macOS / Linux

This command updates the branch, refreshes Python and Node dependencies, verifies resources, exports ONNX graphs, and starts Electron.

```bash
git pull
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend/scripts/export_onnx.py --all
cd electron-app
npm start
```

#### Windows PowerShell

This command updates the branch, refreshes Python and Node dependencies, verifies resources, exports ONNX graphs, and starts Electron.

```powershell
git pull
if (!(Test-Path venv)) { py -m venv venv }
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend/scripts/export_onnx.py --all
cd electron-app
npm start
```

#### Windows Command Prompt

This command updates the branch, refreshes Python and Node dependencies, verifies resources, exports ONNX graphs, and starts Electron.

```bat
git pull
if not exist venv py -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip check
cd electron-app
npm install
npm run verify:resources
cd ..
python backend\scripts\export_onnx.py --all
cd electron-app
npm start
```

### F. Remove Old Native App Builds and Cached Data

Use this section to remove duplicate desktop bundles, stale build outputs, and local app data. These cleanup commands do not install dependencies.

#### macOS

This command removes old macOS build output and scans common app locations for duplicate DepthLens Pro bundles.

```bash
cd electron-app
npm run clean:dist
npm run scan:apps
```

This command removes the user app cache and expects preferences, logs, and caches to reset on next launch.

```bash
rm -rf "$HOME/Library/Application Support/DepthLens Pro"
rm -rf "$HOME/Library/Caches/DepthLens Pro"
```

#### Linux

This command removes Linux build output and expects the next build to recreate `electron-app/dist`.

```bash
cd electron-app
npm run clean:dist
```

This command removes Linux user cache and config directories for DepthLens Pro.

```bash
rm -rf "$HOME/.config/DepthLens Pro"
rm -rf "$HOME/.cache/DepthLensPro"
```

#### Windows PowerShell

This command removes Windows build output and expects the next build to recreate `electron-app\dist`.

```powershell
cd electron-app
npm run clean:dist
```

This command removes Windows user app data for DepthLens Pro.

```powershell
Remove-Item "$env:APPDATA\DepthLens Pro" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\DepthLens Pro" -Recurse -Force -ErrorAction SilentlyContinue
```

#### Windows Command Prompt

This command removes Windows build output and expects the next build to recreate `electron-app\dist`.

```bat
cd electron-app
npm run clean:dist
```

This command removes Windows user app data for DepthLens Pro.

```bat
rmdir /s /q "%APPDATA%\DepthLens Pro"
rmdir /s /q "%LOCALAPPDATA%\DepthLens Pro"
```

### G. Docker Compose

Docker Compose runs the backend and Redis for API-focused development. It does not package or launch the Electron desktop app.

This command builds and starts the API plus Redis services, and the expected outcome is `/live` on `http://127.0.0.1:8765/live`.

```bash
docker compose up --build
```

The Compose file passes `HOST` (`str`, default `0.0.0.0` in Compose for container reachability; local backend config defaults to `127.0.0.1`), `PORT` (`int`, default `8765`), `LOG_LEVEL` (`str`, default `INFO`), `REDIS_HOST` (`str`, default `redis` in Compose and `127.0.0.1` in local config), `REDIS_PORT` (`int`, default `6379`), `REDIS_DB` (`int`, default `0`), `REDIS_PASSWORD` (`str | None`, default empty/`None`), `REDIS_SOCKET_TIMEOUT_SECONDS` (`float`, default `1.5`), `REDIS_MAX_CONNECTIONS` (`int`, default `20`), and `CACHE_TTL_SECONDS` (`int`, default `3600`) into the backend container. The backend also recognizes `DEPTHLENS_AUTO_EXPORT_ONNX` (`bool`, default `false`) if you intentionally want benchmark requests to export missing ONNX graphs.

---

## API Surface

All endpoints are served by the FastAPI app mounted from `backend.api.live` and `backend.api.routes`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Returns service name and API version. |
| `GET` | `/live` | Returns lightweight liveness payload with `status`, `service`, `version`, `pid`, `timestamp`, and `uptime_seconds`. |
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
    └── onnx/
```

---

## Validation & CI

CI and local checks focus on lightweight validation with mocks or stubs where model downloads, GPUs, browsers, Docker, and external services would make tests slow or environment-dependent.

| Check | Source | Purpose |
|---|---|---|
| Python tests | `backend/tests/` | Cache serialization safety, route behavior, ONNX diagnostics, benchmarks, warmup guards, GT handling, and mocked inference flows. |
| Python format/lint/type gates | `pyproject.toml`, `mypy.ini`, `.github/workflows/ci.yml` | Black, Ruff, mypy, and pytest configuration. |
| Electron resource verification | `electron-app/scripts/verify-resources.js` | Confirms packaged resource paths for backend, frontend, and repo-root virtual environment. |
| Electron security policy test | `electron-app/test-security-policy.js` | Validates navigation policy behavior. |
| Packaging scripts | `electron-app/package.json` | Builds ARM-only macOS, Windows, and Linux artifacts; unsupported x64 scripts call `unsupported-arch.js`. |

This command runs the normal Python validation gates in an already prepared environment.

```bash
black --check .
ruff check .
mypy backend/
pytest
```

This command verifies Electron packaging resources from the Electron app directory.

```bash
cd electron-app
npm run verify:resources
```

---

## Troubleshooting

Use the symptom, cause, and resolution entries below for common local issues. Keep `/live` separate from `/ready`: `/live` means the process answers, while `/ready` means required inference imports are available.

### Backend Readiness and Stale Backend Recovery

**Symptom** → The app stays on “Starting engine” or reports no `/live` response. **Cause** → Port `8765` is occupied, the backend process exited before Uvicorn served `/live`, or the configured Python environment cannot import Uvicorn. **Resolution** → Run `cd electron-app && npm run backend:smoke`, then stop any non-DepthLens process that owns port `8765`.

**Symptom** → `/live` returns `ok`, but inference controls stay disabled. **Cause** → `/ready` found a missing required Python module such as `torch`, `cv2`, `PIL`, `fastapi`, `uvicorn`, or `numpy`. **Resolution** → Activate the repo-root virtual environment and run `python -m pip install -r backend/requirements.txt && python -m pip check`.

**Symptom** → Performance Analysis reports ONNX unavailable while PyTorch inference works. **Cause** → ONNX Runtime cannot import, cannot use the requested provider, or cannot load a valid graph from the configured path; benchmarks do not export missing graphs unless `DEPTHLENS_AUTO_EXPORT_ONNX=true` is set. **Resolution** → Run `python backend/scripts/export_onnx.py --all`, then inspect `curl http://127.0.0.1:8765/onnx/status?device=auto`.

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
python backend/scripts/export_onnx.py --all
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

**Symptom** → ONNX Runtime raises `model_path must not be empty`. **Cause** → The selected model did not resolve to an existing non-empty `.onnx` file in the ONNX search path. **Resolution** → Run `python backend/scripts/export_onnx.py --all` and then `curl http://127.0.0.1:8765/onnx/status?device=auto`.

### Troubleshooting: ONNX export failed with SymInt / `torch.export`

**Symptom** → ONNX export fails with `SymInt`, reshape, unflatten, or `torch.export` errors. **Cause** → Some MiDaS/DPT graphs do not export cleanly through every PyTorch ONNX path. **Resolution** → Keep PyTorch inference enabled, inspect the export error, and retry `python backend/scripts/export_onnx.py --all --force` after updating compatible ONNX tooling.

### Verification commands

This command verifies the Python package environment without running tests or downloads beyond installed dependencies.

```bash
python -m pip check
```

This command verifies Electron resource paths before packaging.

```bash
cd electron-app
npm run verify:resources
```
