# DepthLensPro

[![CI/CD Status](https://img.shields.io/badge/CI%2FCD-passing-placeholder.svg)](#validation--ci)
[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](electron-app/package.json)
[![API](https://img.shields.io/badge/API-3.1.0-6f42c1.svg)](backend/main.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Platform](https://img.shields.io/badge/platform-ARM%20desktop%20only-lightgrey.svg)](#supported-platforms)

Last Updated: 10th June 2026

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

- Monocular depth estimation using MiDaS-family models: `MiDaS_small`, `DPT_Hybrid`, and `DPT_Large`.
- ONNX Runtime execution when exported static graphs are available, with automatic first-benchmark ONNX graph export, explicit diagnostics for missing weights, missing runtime imports, provider fallback, and CPU-only execution.
- Single-image and batch inference, with batch requests capped at **10 images**.
- Supported source uploads: `PNG`, `JPG/JPEG`, `WEBP`, and `BMP`.
- Optional Ground Truth mode for one source image plus one GT depth file (`PNG`, `TIFF`, or `NPY`, **20 MB** max) to compute benchmark metrics without changing the standard image-only workflow.
- Selective output generation for colorized and/or grayscale depth maps so interactive workspace requests avoid unnecessary encoding.
- Colormap options: `inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, and `turbo`.
- Runtime model comparison across supported architectures with side-by-side outputs and metric charts.
- Experiment Workspace for named local validation runs, side-by-side RGB/prediction/GT/error previews, and JSON/CSV exports.
- Prediction statistics, proxy/self-consistency diagnostics, and true GT benchmark metrics are separated so proxy values are not presented as standard accuracy metrics.

### System Architecture & Observability

- FastAPI ASGI backend with explicit route, inference, cache, benchmark, configuration, and hardware modules.
- Redis-backed response cache plus normalized-depth reuse keyed by image, model, resolved device, and max dimension, with TTL control and in-memory fallback.
- Structured runtime configuration through environment variables and optional `.env` values.
- JSON structured logging for collector-friendly backend logs.
- Lightweight liveness, inference readiness, ONNX status, device inventory, health diagnostics, benchmark, cache, memory, disk, and acceleration signals exposed through API endpoints.
- Docker and Docker Compose support for backend-plus-Redis execution.

### Desktop Client

- Electron desktop shell with automatic FastAPI child-process lifecycle management.
- Sandboxed renderer, `contextIsolation` enabled, `nodeIntegration` disabled, and navigation constrained to loopback origins.
- Splash screen while Electron waits for lightweight `/live`, followed by progressive renderer startup: `/live` shows the backend is reachable, `/ready` gates inference controls, and `/health`, `/devices`, plus cache metrics load afterward with developer timing logs.
- Drag-and-drop upload workflow with progress, adaptive ETA, cancellation, optional GT pairing, result cards, lightbox metrics, and download actions.
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

Desktop packaging targets ARM platforms only. Apple Silicon macOS runs PyTorch MPS inference through Metal without Rosetta translation overhead; Windows ARM and Linux ARM keep CPU plus any available CUDA/XPU runtime paths provided by the installed Python environment. ONNX Runtime remains available as an alternative execution path when static model graphs and compatible providers are present.

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
| API service | Python `3.10`–`3.12`, FastAPI, Uvicorn | ASGI API, request validation, lifecycle hooks, diagnostics |
| AI runtime | PyTorch, Torch Hub, ONNX, ONNX Runtime | MiDaS model loading and accelerated inference |
| Image processing | OpenCV Headless, NumPy, Pillow | Input normalization, depth-map post-processing, PNG encoding without desktop GUI library requirements |
| Cache | Redis with in-memory fallback | TTL-based reuse of repeated inference results |
| Configuration | `pydantic-settings` with fallback shim | Environment and `.env` driven backend configuration |
| Observability | JSON logging, `/live`, `/ready`, `/health`, `/devices`, `/onnx/status`, `/benchmark`, `/cache/metrics` | Liveness, inference readiness, ONNX provider/weight diagnostics, runtime state, resource telemetry, benchmark reporting |
| Delivery | Docker, Docker Compose, GitHub Actions | Containerized backend, Redis stack, quality gates |

---

## Observability & Telemetry

DepthLensPro exposes operational signals suitable for local diagnostics, release validation, and production-style monitoring.

| Signal | Surface | Description |
|---|---|---|
| Structured logs | Backend stdout | JSON records with timestamp, level, logger, module, function, line, process, thread, and exception details. |
| Liveness | `GET /live` | Immediate backend availability check used by Electron and the renderer; does not load models, probe accelerators, query Redis, or gather telemetry. |
| Inference readiness | `GET /ready` | Fast dependency and packaging check used by the renderer before enabling inference; imports required runtime modules, reports optional Redis/ONNX availability, lists ONNX weight paths, and does not load MiDaS weights. |
| Health diagnostics | `GET /health` | Full diagnostics including primary device, loaded models, PyTorch version, cached acceleration checks, device inventory, cache summary, timing fields, warmup state, and system metadata. Degraded diagnostics do not make inference unavailable. |
| Device inventory | `GET /devices` | Lightweight device list used after `/live`; always includes CPU and uses cached discovery where possible. |
| Memory telemetry | `GET /health` | Memory status, pressure percentage, limit threshold, total bytes, available bytes, and used bytes. |
| Disk telemetry | `GET /health` | Disk status, monitored path, usage percentage, limit threshold, total bytes, free bytes, and used bytes. |
| Cache metrics | `GET /cache/metrics` and `GET /health` | Redis availability, backend type, hit/miss counts, fallback failures, keyspace size, and TTL. |
| Runtime benchmarks | `GET /benchmark` and `GET /api/benchmark` | PyTorch versus ONNX Runtime latency, throughput, memory, speedup comparison, automatic missing-graph export when enabled, and explicit ONNX states (`available`, `missing_weights`, `onnxruntime_missing`, `provider_unavailable`, `export_failed`, `runtime_error`, `unavailable`). |
| ONNX diagnostics | `GET /onnx/status`, `/ready`, and `/health` | Supported model IDs, expected weight paths, file sizes, ONNX Runtime import status, available providers, selected provider/fallback, and export commands. |
| Session analytics | Desktop workspace | Processed image count, latency history, average/min/max latency, throughput, total inference time, cache hits, and error count. |
| Experiment validation | Desktop Experiments panel | Named local runs, image/GT metadata, latency, metrics, warnings, side-by-side previews, error heatmaps when GT is valid, and JSON/CSV exports. |

---

## Supported Platforms

DepthLens Pro now supports **ARM desktop targets only**. Normal users should open the native app; they should not manually start FastAPI or Uvicorn. Electron owns backend startup, liveness checks through `/live`, renderer readiness checks through `/ready`, and backend shutdown.

| Status | Platform | Architecture | Notes |
|---|---|---|---|
| Supported | macOS Apple Silicon only | `darwin arm64` | Build/open `electron-app/dist/mac-arm64/DepthLens Pro.app` or install one `/Applications/DepthLens Pro.app` from the Apple Silicon DMG. |
| Supported | Windows ARM only | `win32 arm64` | Build with `npm run build:win:arm64`; Windows x64 installers are intentionally not produced. |
| Supported | Linux ARM only | `linux arm64` / `aarch64` | Build with `npm run build:linux:arm64`; Linux x64 artifacts are intentionally not produced. |
| Unsupported | Intel Mac / macOS x64 | `darwin x64` | Blocked at Electron startup before backend launch. |
| Unsupported | macOS universal builds | universal | Unsupported because they can create duplicate/stale bundles and Intel launch paths. |
| Unsupported | Windows x64 | `win32 x64` | Unsupported build scripts fail with a clear architecture message. |
| Unsupported | Linux x64 | `linux x64` | Unsupported build scripts fail with a clear architecture message. |

The bundle identifier remains stable as `com.ayushmanraha.depthlens-pro`, and the product name remains exactly `DepthLens Pro` without architecture suffixes.

The runtime guard shows: “DepthLens Pro currently supports Apple Silicon macOS, Windows ARM, and Linux ARM only.” It also includes the detected platform and architecture and exits before starting the backend.

## Quick Start / Runbook

DepthLensPro uses a local FastAPI backend at `http://127.0.0.1:8765` by default. The Electron app starts that backend automatically in development and packaged desktop modes, waits for `GET /live`, then exposes the resolved URL to the renderer through the secure preload bridge. The renderer calls `GET /ready` before enabling inference controls so missing OpenCV/PyTorch/packaging dependencies are surfaced clearly. `GET /health` is intentionally reserved for fuller diagnostics and may be degraded without making inference unavailable.

> **First run:** model weights may be downloaded through `torch.hub` the first time a model is lazily loaded. Subsequent runs use the local model cache. Optional background warmup is available with `DEPTHLENS_PRELOAD_MODEL=true`, `DEPTHLENS_WARMUP_MODEL=MiDaS_small`, and `DEPTHLENS_WARMUP_DEVICE=auto`; warmup failure does not break `/live` or lazy inference.
>
> **Cache note:** Redis is optional. If Redis is unavailable or not installed, the backend logs the condition and falls back to an in-memory cache automatically.
>
> **Repeatable dependency note:** Always run the backend dependency install/verify step before launching, testing, or building. This is safe to repeat and ensures new or updated backend requirements are installed after a fresh clone, `git pull`, branch switch, or code update. Continue only after `python -m pip check` finishes without errors. Performance Analysis also requires ONNX export/runtime packages (`onnx`, `onnxruntime`, and `onnxscript`), which are installed from `backend/requirements.txt`; missing `.onnx` graph files are exported automatically the first time a benchmark is run unless `DEPTHLENS_AUTO_EXPORT_ONNX=false`.

### Prerequisites

- Python `3.10`–`3.12` recommended for PyTorch, OpenCV, and ONNX Runtime compatibility
- `pip`
- Node.js and npm
- Git
- Optional: Redis for distributed cache validation

### A. Get the Repository in Downloads

Keep the working copy under the user Downloads folder on every supported platform. The repository URL is shown as a placeholder because this checkout does not declare a Git remote.

#### macOS / Linux

```bash
cd ~/Downloads
git clone https://github.com/<your-org-or-user>/DepthLensPro.git
cd DepthLensPro
```

#### Windows PowerShell

```powershell
cd "$env:USERPROFILE\Downloads"
git clone https://github.com/<your-org-or-user>/DepthLensPro.git
cd DepthLensPro
```

#### Windows Command Prompt

```bat
cd %USERPROFILE%\Downloads
git clone https://github.com/<your-org-or-user>/DepthLensPro.git
cd DepthLensPro
```

### B. Install and Verify Backend Dependencies

DepthLensPro stores Python backend requirements in `backend/requirements.txt`. The virtual environment is created at the repository root as `venv/` because packaged Electron builds copy `../venv` into the native app resources. Run these commands from inside `backend/` every time before a terminal launch, backend-only launch, native build, native test, rebuild, or run after pulling changes.

#### macOS / Linux

```bash
cd ~/Downloads/DepthLensPro

cd backend
python3 -m venv ../venv
source ../venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..
```

#### Windows PowerShell

```powershell
cd "$env:USERPROFILE\Downloads\DepthLensPro"

cd backend
py -3 -m venv ..\venv
..\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..
```

If PowerShell blocks virtual environment activation, run this once in the same PowerShell profile and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

#### Windows Command Prompt

```bat
cd %USERPROFILE%\Downloads\DepthLensPro

cd backend
py -3 -m venv ..\venv
..\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..
```

### C. Native App Build / Install by Platform

Run the backend dependency install/verify commands in section B first, then install Electron dependencies from `electron-app/` and build the supported ARM package. Native build and test commands do **not** replace the backend dependency step; repeat section B after `git pull`, branch switching, dependency file changes, or before rebuilding a newer app version.

#### macOS Native App Build

Builds unsigned macOS artifacts for **Apple Silicon (`arm64`) only**. `npm run build:mac` cleans old packaged outputs first, builds the Apple Silicon target, and scans for duplicate app bundles. The intended local Apple Silicon build output is:

```text
electron-app/dist/mac-arm64/DepthLens Pro.app
```

```bash
cd ~/Downloads/DepthLensPro

cd backend
python3 -m venv ../venv
source ../venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm run verify:resources
npm run build:mac
```

Open either the local app bundle at `electron-app/dist/mac-arm64/DepthLens Pro.app` for testing or the generated Apple Silicon DMG from `electron-app/dist/`, drag **one** **DepthLens Pro** app to `/Applications`, and launch it. The `dist/` app is build output; `/Applications/DepthLens Pro.app` is the installed copy. Keeping both can make Spotlight show two icons until one is removed and indexing catches up. If macOS Gatekeeper blocks an unsigned local build, clear quarantine metadata:

```bash
xattr -cr "/Applications/DepthLens Pro.app"
```

#### Windows Native App Build

Builds a **Windows ARM64 only** NSIS installer. Run these commands from PowerShell so the Windows virtual environment layout is used (`venv\Scripts\python.exe`). Windows x64 builds are unsupported and intentionally fail.

```powershell
cd "$env:USERPROFILE\Downloads\DepthLensPro"

cd backend
py -3 -m venv ..\venv
..\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm run verify:resources
npm run build:win:arm64
```

Run the generated installer from `electron-app\dist\`, then launch **DepthLens Pro** from the Start menu or desktop shortcut. Packaged builds expect the prepared `venv/`, `backend/`, and `frontend/` resources to be present in the installer resources.

#### Linux ARM Native App Build

Builds a **Linux ARM64/aarch64 only** AppImage. Prepare and verify the repo-root virtual environment first so electron-builder can copy `backend/`, `frontend/`, and `venv/` into packaged resources.

```bash
cd ~/Downloads/DepthLensPro

cd backend
python3 -m venv ../venv
source ../venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm run verify:resources
npm run build:linux:arm64
```

Run the generated ARM64 AppImage from `electron-app/dist/`. Linux x64 builds are intentionally unsupported.

#### Native App Development Test

Use this when you want to test the native shell without creating an installer. Run section B first, then:

```bash
cd ~/Downloads/DepthLensPro/electron-app
npm install
npm run verify:resources
npm start
```

On Windows PowerShell, use the same commands after entering `cd "$env:USERPROFILE\Downloads\DepthLensPro\electron-app"`.

#### End-User Installation from Release Artifacts

- **macOS:** Download the latest `DMG`, mount it, drag **DepthLens Pro** into `/Applications`, and launch the app.
- **Windows ARM:** Download the latest Windows ARM installer (`.exe`), run it, and launch **DepthLens Pro** from the Start menu or desktop shortcut. Windows x64 is unsupported.
- **Linux ARM:** Build or download the ARM64/aarch64 AppImage and run one copy of **DepthLens Pro**. Linux x64 is unsupported.

### D. Terminal-Only Local Test / Development

Use this path when you want to run the full app from terminals without creating a native installer. Repeat the backend dependency install/verify step before every launch, especially after pulling changes or switching branches.

The Electron development command is the simplest full-app terminal workflow: one terminal runs Electron, and Electron starts FastAPI on `127.0.0.1:8765` automatically.

#### macOS / Linux

```bash
cd ~/Downloads/DepthLensPro

cd backend
python3 -m venv ../venv
source ../venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm start
```

#### Windows PowerShell

```powershell
cd "$env:USERPROFILE\Downloads\DepthLensPro"

cd backend
py -3 -m venv ..\venv
..\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm start
```

#### Windows Command Prompt

```bat
cd %USERPROFILE%\Downloads\DepthLensPro

cd backend
py -3 -m venv ..\venv
..\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm start
```

#### Backend and Static Frontend in Separate Windows

Use two terminals when validating API routes separately from the renderer. Terminal 1 starts the backend from the repository root after section B has passed:

```bash
cd ~/Downloads/DepthLensPro
source venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

For Windows PowerShell, use `cd "$env:USERPROFILE\Downloads\DepthLensPro"` and `.\venv\Scripts\Activate.ps1` before the same `python -m uvicorn ...` command. For Windows Command Prompt, use `cd %USERPROFILE%\Downloads\DepthLensPro` and `venv\Scripts\activate.bat`.

Terminal 2 can open `frontend/index.html` directly in a browser or use the Electron development command from `electron-app/` if you want the native shell. If opening the static frontend directly, start the backend first and optionally set `localStorage.depthlens_api_url` to `http://127.0.0.1:8765`.

```bash
cd ~/Downloads/DepthLensPro/frontend
open index.html      # macOS
xdg-open index.html  # Linux
```

```powershell
cd "$env:USERPROFILE\Downloads\DepthLensPro\frontend"
start index.html
```

```bat
cd %USERPROFILE%\Downloads\DepthLensPro\frontend
start index.html
```

Liveness, diagnostics, and route checks:

```bash
curl http://127.0.0.1:8765/
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/devices
curl http://127.0.0.1:8765/health
```

From `electron-app/`, you can also run the npm smoke helper while the backend is running:

```bash
npm run backend:live
npm run backend:ready
npm run backend:devices
npm run backend:health
npm run backend:smoke
npm run verify:resources
```

### E. Update, Rebuild, and Run the Latest Code

After every `git pull`, branch switch, or dependency file change, install and verify backend dependencies again from `backend/` before launching, testing, or building. Then install/update Electron dependencies from `electron-app/`.

#### macOS / Linux

```bash
cd ~/Downloads/DepthLensPro
git pull

cd backend
if [ ! -d ../venv ]; then python3 -m venv ../venv; fi
source ../venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm run verify:resources

# Run the updated code from terminal:
npm start

# Or rebuild a new native app version instead:
npm run build:mac        # macOS Apple Silicon
npm run build:linux:arm64 # Linux ARM64/aarch64
```

#### Windows PowerShell

```powershell
cd "$env:USERPROFILE\Downloads\DepthLensPro"
git pull

cd backend
if (!(Test-Path "..\venv")) { py -3 -m venv ..\venv }
..\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm run verify:resources

# Run the updated code from terminal:
npm start

# Or rebuild a new native app version instead:
npm run build:win:arm64
```

#### Windows Command Prompt

```bat
cd %USERPROFILE%\Downloads\DepthLensPro
git pull

cd backend
if not exist ..\venv py -3 -m venv ..\venv
..\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

cd electron-app
npm install
npm run verify:resources

REM Run the updated code from terminal:
npm start

REM Or rebuild a new native app version instead:
npm run build:win:arm64
```

### F. Remove Old Native App Builds and Cached Data

Use this before a fresh native install when old build output, stale app bundles, caches, logs, or backend pid files may conflict with a new build. These commands remove app settings, logs, caches, and local DepthLens Pro runtime data. Back up any user-created exports or files stored in those app data folders before running cleanup.

#### macOS

```bash
cd ~/Downloads/DepthLensPro

rm -rf electron-app/dist electron-app/build electron-app/out electron-app/release electron-app/.tauri electron-app/.electron
rm -rf frontend/dist frontend/build
rm -rf backend/__pycache__ backend/.pytest_cache

rm -rf "/Applications/DepthLens Pro.app"
rm -rf "$HOME/Applications/DepthLens Pro.app"
rm -rf "$HOME/Library/Application Support/DepthLens Pro"
rm -rf "$HOME/Library/Caches/DepthLens Pro"
rm -rf "$HOME/Library/Logs/DepthLens Pro"
rm -f "$HOME/Library/Preferences/com.ayushmanraha.depthlens-pro.plist"
```

You can also run `npm run clean:dist`, `npm run clean:install`, and `npm run scan:apps` from `electron-app/` for the macOS build-output and duplicate-app cleanup helpers that are already included in this repository.

#### Linux

```bash
cd ~/Downloads/DepthLensPro

rm -rf electron-app/dist electron-app/build electron-app/out electron-app/release electron-app/.tauri electron-app/.electron
rm -rf frontend/dist frontend/build
rm -rf backend/__pycache__ backend/.pytest_cache

rm -rf "$HOME/.config/DepthLens Pro"
rm -rf "$HOME/.cache/DepthLens Pro"
rm -rf "$HOME/.local/share/DepthLens Pro"
rm -rf "$HOME/.local/state/DepthLens Pro"
```

#### Windows PowerShell

```powershell
cd "$env:USERPROFILE\Downloads\DepthLensPro"

Remove-Item -Recurse -Force .\electron-app\dist, .\electron-app\build, .\electron-app\out, .\electron-app\release, .\electron-app\.tauri, .\electron-app\.electron -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\frontend\dist, .\frontend\build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\backend\__pycache__, .\backend\.pytest_cache -ErrorAction SilentlyContinue

Remove-Item -Recurse -Force "$env:APPDATA\DepthLens Pro" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\DepthLens Pro" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Temp\DepthLens Pro" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:USERPROFILE\AppData\Local\Programs\DepthLens Pro" -ErrorAction SilentlyContinue
```

#### Windows Command Prompt

```bat
cd %USERPROFILE%\Downloads\DepthLensPro

rmdir /s /q electron-app\dist 2>nul
rmdir /s /q electron-app\build 2>nul
rmdir /s /q electron-app\out 2>nul
rmdir /s /q electron-app\release 2>nul
rmdir /s /q electron-app\.tauri 2>nul
rmdir /s /q electron-app\.electron 2>nul
rmdir /s /q frontend\dist 2>nul
rmdir /s /q frontend\build 2>nul
rmdir /s /q backend\__pycache__ 2>nul
rmdir /s /q backend\.pytest_cache 2>nul

rmdir /s /q "%APPDATA%\DepthLens Pro" 2>nul
rmdir /s /q "%LOCALAPPDATA%\DepthLens Pro" 2>nul
rmdir /s /q "%LOCALAPPDATA%\Temp\DepthLens Pro" 2>nul
rmdir /s /q "%USERPROFILE%\AppData\Local\Programs\DepthLens Pro" 2>nul
```

### G. Docker Compose

Docker Compose is available for backend-plus-Redis validation. It does not replace the native desktop packaging flow.

```bash
cd ~/Downloads/DepthLensPro
docker compose up --build
```

---

## API Surface

Base URL: `http://127.0.0.1:8765`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API metadata and service version. |
| `GET` | `/live` | Lightweight liveness signal used by Electron/frontend startup; returns immediately without model loading, accelerator probing, Redis, or telemetry. |
| `GET` | `/ready` | Inference readiness check: required Python runtime modules, optional Redis/ONNX modules, Python/torch platform details, static model metadata, and ONNX weight paths without loading MiDaS weights. |
| `GET` | `/health` | Full diagnostics: devices, cached acceleration checks, telemetry, cache metrics, warmup status, timing fields, and runtime metadata. Degraded diagnostics do not imply backend offline. |
| `GET` | `/devices` | Lightweight discovered compute targets with type, compute class, hardware details, primary device, and CPU fallback. |
| `GET` | `/models` | Supported model registry with runtime notes. |
| `GET` | `/colormaps` | Supported colormap keys. |
| `POST` | `/estimate` | Single-image depth estimation. Form fields: `file`, `model`, `colormap`, `device`, optional `metrics` (`none`, `fast`, `full`), optional `outputs` (`color`, `gray`, `color,gray`), and optional `max_dim`. |
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
  "diagnostics_status": "ok",
  "version": "3.1.0",
  "primary_device": "mps",
  "devices": { "mps": { "available": true, "type": "GPU" } },
  "loaded_models": ["MiDaS_small:mps"],
  "timings_ms": { "device_discovery": 1.2, "accelerator_probe": 2.4, "cache_metrics": 0.8, "health_generation": 5.6 },
  "warmup": { "enabled": false, "model": "MiDaS_small", "device": "auto" },
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
    "entropy": 6.82,
    "dynamic_range": 8.14,
    "prediction_stats": { "mean": 0.51, "std": 0.18, "entropy": 6.82 },
    "proxy_metrics": { "ssim": 0.812, "silog": 7.43, "psnr": 28.5 },
    "gt_metrics": { "abs_rel": 0.124, "gt_rmse": 0.42, "delta_1": 0.84 },
    "unavailable": { "lpips": "not_implemented" },
    "warnings": ["Resized GT to prediction resolution using nearest-neighbor alignment."]
  },
  "gt_metadata": {
    "provided": true,
    "format": "png",
    "original_shape": [768, 1024],
    "aligned_shape": [768, 1024],
    "valid_pixel_count": 786000,
    "invalid_pixel_count": 432,
    "alignment_policy": "gt_resize_to_prediction:none",
    "scale_alignment": "median_scale"
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
| `min`, `max`, `mean`, `std`, `median`, `entropy`, `dynamic_range`, `coverage` | Prediction statistics | Distribution summaries of the normalized predicted depth map. These do not require GT and are not benchmark accuracy metrics. |
| `ssim` | Proxy/self-consistency | **RGB–Depth Structural Proxy** comparing grayscale input structure to predicted depth. This is not reference SSIM. |
| `silog` | Proxy/self-consistency | **Log-Depth Dispersion Proxy** computed from prediction-only log-depth variation. True SILog requires GT. |
| `psnr` | Proxy/self-consistency | **Depth Variance PSNR Proxy** relative to the predicted depth mean. True PSNR requires a reference/GT map. |
| `mae` / `rmse` / `log_rmse` | Proxy/self-consistency | **Mean Absolute Deviation from Predicted Mean**, **RMS Deviation from Predicted Mean**, and **Log-Depth Deviation**. These are not standard GT MAE/RMSE/Log RMSE. |
| `gradient_mean` / `gradient_std` / `gradient_error` / `edge_density` | Prediction geometry/proxy | Sobel-gradient statistics over the predicted depth map. `gradient_error` is displayed as **Depth Edge Proxy** because it is not a GT error. |
| `abs_rel`, `sq_rel`, `gt_mae`, `gt_rmse`, `gt_log_rmse`, `delta_1/2/3` | Ground truth benchmark metrics | Computed only when a valid optional GT file is uploaded. MiDaS relative depth is median-scale aligned to valid GT pixels before metric calculation. |
| `lpips`, `ordinal_error`, `surface_normal_error`, `gt_ssim`, `gt_psnr` | Ground truth required / not implemented | Reported with explicit unavailable reasons until implemented safely. |

### Ground Truth Mode and Metric Alignment

Ground Truth upload is optional. Standard image-only generation and multi-image queues remain unchanged when the Workspace toggle is off. When **Upload ground truth data** is enabled, DepthLensPro processes exactly one source image plus one GT file at a time and blocks generation until both are selected.

Supported GT formats are `.png`, `.tif`/`.tiff`, and `.npy` with a **20 MB** limit. Single-channel depth is preferred. Multi-channel PNG/TIFF files are converted to grayscale with a warning; multi-channel NPY arrays are rejected unless shaped `H×W×1`. EXR, PFM, PDF reports, and 16-bit predicted-depth export are not implemented in this pass to avoid non-portable or heavy dependencies.

For shape mismatches, the backend resizes GT to the prediction resolution with nearest-neighbor sampling. Because MiDaS predicts relative depth, GT metrics use median scale alignment between valid predicted pixels and valid GT pixels. Invalid, non-finite, zero, or negative GT pixels are masked. Responses include `gt_metadata`, metric warnings, optional `gt_depth_map`, and an `error_heatmap` PNG when GT is valid.

### Experiment Workspace

The Experiments panel provides a lightweight local validation workflow. Users name a run, execute the current Workspace queue, inspect a results table/leaderboard, view RGB/predicted/GT/error side-by-side previews where available, and export JSON or CSV reports. Exports are browser/Electron-safe local downloads and work across supported ARM desktop targets without cloud services or OS-specific paths. The panel uses the same theme tokens as the rest of DepthLensPro, so dark and light modes remain supported.

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
│   ├── model_metadata.py            # Static model/colormap metadata safe for lightweight imports
│   ├── api/routes.py                # HTTP route handlers
│   ├── services/
│   │   ├── benchmarks.py            # Runtime benchmark service
│   │   ├── cache_service.py         # Redis cache and memory fallback
│   │   ├── diagnostics.py           # Readiness and dependency diagnostics
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

Run the backend dependency install/verify step from `backend/` before local validation, then run the quality gates from the repository root:

```bash
cd ~/Downloads/DepthLensPro
cd backend
source ../venv/bin/activate 2>/dev/null || { python3 -m venv ../venv && source ../venv/bin/activate; }
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
cd ..

black --check .
ruff check .
mypy backend/
pytest
```

On Windows, use the matching activation command from section B before the same `python -m pip ...`, `black`, `ruff`, `mypy`, and `pytest` checks. The GitHub Actions pipeline runs the same core checks for formatting, linting, typing, and tests. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution requirements.

---

## Troubleshooting

| Issue | Recommended Action |
|---|---|
| Backend imports fail or `ModuleNotFoundError` appears | Reinstall and verify backend dependencies from `~/Downloads/DepthLensPro/backend` using section B. Do not install backend dependencies from `frontend/`, `electron-app/`, or another subfolder. Activate the repo-root `venv/` before backend commands, repeat this after `git pull`, branch switching, or dependency file changes, then launch from the repo/resources root with `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765`. |
| Frontend reports that the engine is offline | Confirm the backend answers `curl http://127.0.0.1:8765/live`; if `/live` fails, Electron did not start the backend or the selected port is wrong. |
| Frontend reports that the engine is degraded | Confirm `curl http://127.0.0.1:8765/ready`; required modules such as `cv2`, `torch`, `numpy`, FastAPI, Uvicorn, and Pillow must import successfully before inference controls are enabled. |
| Uvicorn is running but app still reports offline | Verify the port matches `DEPTHLENS_BACKEND_PORT` and that `/live` returns `service: "DepthLens Pro API"`; a different service on the port is treated as a conflict. |
| Device selector stuck on Auto | Check `curl http://127.0.0.1:8765/devices`; the endpoint should return at least CPU after `/live` succeeds. |
| Browser CORS or local file issue | The native app intentionally loads the renderer from `file://` and the backend allows loopback API calls. If opening `frontend/index.html` directly in a browser, start the backend manually and optionally set `localStorage.depthlens_api_url`. |
| Slow inference | Use `MiDaS_small`, select an available accelerator, keep default workspace `metrics=fast`/`outputs=color`, reduce `DEPTHLENS_MAX_DIM`, or export ONNX weights for the target model. |
| Port `8765` already in use | If the existing service is a live DepthLens backend, Electron reuses it. If the port is occupied by another process and `DEPTHLENS_BACKEND_PORT` is not pinned, Electron tries a nearby free loopback port and passes that resolved URL to the renderer. |
| First run or first inference is slow | Allow lazy model loading and initial model-weight download to complete; later runs use the local Torch cache. Enable optional background warmup with `DEPTHLENS_PRELOAD_MODEL=true` if desired. |
| macOS blocks launch | Run `xattr -cr "/Applications/DepthLens Pro.app"` for unsigned local builds. |
| Packaged backend does not start | Check the desktop logs for backend URL, Python path, cwd, command, exit code/signal, backend output tail, and resource existence. Run `npm run verify:resources` from `electron-app/` before packaging or pass a packaged resources directory to `node scripts/verify-resources.js <resources-root>`. |
| Python not found in packaged app | Verify the repo-root `venv/` folder exists before packaging and is included in electron-builder `extraResources`. |
| Packaged resources missing | Run `npm run verify:resources`; it validates `backend/`, `backend/app.py`, `frontend/`, `frontend/index.html`, and platform Python paths (`venv/bin/python3`, `venv/bin/python`, or `venv/Scripts/python.exe`). |
| Duplicate app icon on macOS | Run `npm run scan:apps`. Remove duplicate bundles with `npm run clean:install` and `npm run clean:dist`, then open only `dist/mac-arm64/DepthLens Pro.app` or one installed `/Applications/DepthLens Pro.app`. Spotlight can take time to update. |
| `acceleration_ok: false` | A GPU backend failed the runtime probe; CPU inference remains available. |
| `/health` reports degraded status | Inspect memory pressure and disk usage telemetry for threshold violations. |
| Redis cache unavailable | Verify Redis host, port, credentials, and container health; the backend uses in-memory fallback automatically. |
| Missing ONNX weights | Performance Analysis now exports the selected model automatically on the first benchmark run when `DEPTHLENS_AUTO_EXPORT_ONNX` is unset or true. To pre-generate graphs manually, run `python backend/scripts/export_onnx.py --all` after installing `backend/requirements.txt`, or set `DEPTHLENS_ONNX_DIR` to the directory containing `.onnx` files. ONNX weights are generated artifacts and are not committed. |
| `/benchmark` reports ONNX unavailable | Check `curl http://127.0.0.1:8765/onnx/status?device=auto`. The response distinguishes `missing_weights`, `onnxruntime_missing`, `provider_unavailable`, `export_failed`, `runtime_error`, and CPU/provider fallback. If automatic export fails, verify that `onnx`, `onnxruntime`, `onnxscript`, `torch`, `torchvision`, and `timm` were installed by repeating section B. PyTorch benchmark/inference can still run. |
| ONNX uses CPU fallback | The installed ONNX Runtime package does not expose the preferred provider for the selected device. This is valid and platform-neutral; install a provider-enabled runtime locally if acceleration is required. |
| GT upload rejected | Ensure the file is PNG, TIFF, or NPY, under 20 MB, numeric/single-channel where required, and contains finite positive valid pixels. Standard image-only inference remains unaffected. |
| GT metrics look scale-adjusted | MiDaS predicts relative depth. DepthLensPro resizes GT to prediction resolution and applies median scale alignment before Abs Rel, Sq Rel, MAE, RMSE, Log RMSE, and δ metrics. |

### Backend Readiness and Stale Backend Recovery

- `/live` is the lightweight startup endpoint. It must answer quickly once Uvicorn accepts requests and does not load models, probe hardware, query Redis, or collect disk/memory telemetry.
- `/devices` returns device inventory plus `primary_device`; the frontend uses that primary device for Auto while preserving CPU and available accelerators such as MPS, CUDA, or XPU.
- `/health` is diagnostics only. It can be degraded or slower without making generation unavailable. Generation depends on the backend being online and queued files, not `/health` success.
- Opening the native app twice is safe: Electron uses a single-instance lock, focuses the first window, and the second instance does not start another backend.

If a stale Python backend is stuck on port `8765`, `curl` may connect but receive 0 bytes until it times out. Diagnose and clean it with:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
npm run kill:backend
npm run smoke:backend
```

`npm run kill:backend` uses the pid file and/or port owner, prints the detected PID and command line, and kills only safe DepthLens backend matches. If manual termination is needed, use the actual PID number shown by diagnostics. For example, if the PID is `66208`, run:

```bash
kill -9 66208
```

Do not type placeholders such as `kill -9 <PID>` or `kill -9 THE_NUMBER_SHOWN`; replace them with the real number. If a non-DepthLens process owns port `8765`, the app does not kill it automatically and shows the detected PID plus an exact command using that PID.

### Acceptance Checks

While the app is open:

```bash
curl --max-time 3 -v http://127.0.0.1:8765/live
curl --max-time 5 -v http://127.0.0.1:8765/devices
```

After quitting the app:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

The final `lsof` command should print nothing.

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
