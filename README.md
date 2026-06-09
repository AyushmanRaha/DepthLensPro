# DepthLensPro
**AI-Powered 2D to 3D Depth Map Generation Desktop Application**

**Last Updated:** 9th June 2026

DepthLensPro is an intelligent native cross-platform desktop application built with Electron that converts 2D images into high-quality depth visualizations using monocular depth estimation (MiDaS family). It combines a FastAPI inference backend (**API v3.1.0**, desktop package **v4.0.0**) with a production-style browser workspace for batch processing, device-aware execution, model comparison, runtime analytics, Redis-backed caching, and ONNX Runtime acceleration when static weights are available.

---

## Features

- AI-powered monocular depth estimation with multiple MiDaS models (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`)
- ONNX Runtime inference path with automatic PyTorch fallback when exported static weights are unavailable
- Native cross-platform desktop application via **Electron v42** with automatic backend lifecycle management
- **Liquid-metal animated welcome screen** (v5.0): canvas blob animation — drops spawn from ring → fall toward centre → merge → morph into particle cloud → crystallise into final gradient text
- **Vector background canvas** on the welcome screen — grid lines, diagonal accents, circuit connector segments, and intersection dots; theme-aware and re-drawn on resize
- Persistent dark/light **theme toggle** — migrates from landing screen corner to header slot via spring-physics `getBoundingClientRect` delta animation on workspace entry
- Batch processing support (up to **10 images** per request)
- Broad image format compatibility (PNG, JPG/JPEG, WEBP, BMP)
- Dual output rendering (colorized depth map + grayscale depth map)
- Rich colormap support (`inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, `turbo`)
- Compute-device routing with `Auto` and explicit `CPU`, `CUDA`, `MPS`, and `XPU` selection
- Device filter toggle (All / CPU / GPU / NPU) dynamically generated from detected hardware
- Expanded MDE metric suite (SSIM, SILog, PSNR, entropy, dynamic range, edge/gradient statistics, histogram spread, and consistency proxies)
- Dedicated model comparison panel with side-by-side output cards and metric-driven charting
- Session analytics dashboard (latency trends, throughput, cache hits, error counts, total inference time)
- Redis-backed inference cache for repeated image + model + colormap + device combinations, with TTL controls and in-memory fallback
- Live cache telemetry via `/cache/metrics` (backend type, hit/miss counts, keyspace size, Redis availability, fallback failures, TTL)
- PyTorch vs ONNX Runtime benchmark endpoint (`/benchmark` and `/api/benchmark`) with latency, throughput, memory, and speedup comparison data
- **System telemetry** in `/health` — memory pressure and disk utilisation reported alongside engine status
- **Structured JSON logging** — all backend log records emitted as JSON via `JsonLogFormatter` for production collector compatibility
- **Structured runtime configuration** via `backend/config.py` — `pydantic-settings`-backed `Settings` class reads `HOST`, `PORT`, `LOG_LEVEL`, `DEBUG`, Redis connection settings, and cache TTL from environment and `.env` file
- API-first backend architecture for integration, automation, and deployment workflows
- Per-request upload cancel support with ETA/progress visibility
- Persistent user preferences (model, colormap, device) via `localStorage`

---

## Desktop Application

- DepthLens Pro ships as a native desktop application built with **Electron v42** — a double-clickable `.app` on macOS and `.exe` on Windows.
- The FastAPI inference backend starts and stops automatically as a child process — no manual uvicorn command needed.
- The app targets **arm64 (Apple Silicon)** natively, running PyTorch MPS inference via Metal with no Rosetta translation overhead.
- A splash screen is displayed while the backend warms up. Main window appears only after the engine is ready (or a graceful timeout).
- Distributed as a **DMG** for macOS. No internet connection required after first model weight download.
- **Security:** `contextIsolation` enabled, `nodeIntegration` disabled, renderer sandboxed. Backend only accessible on `127.0.0.1`.
- macOS traffic-light buttons use `titleBarStyle: "hidden"` with `trafficLightPosition` for correct header layout. The preload exposes `platform` and `arch` via `contextBridge` so the renderer can conditionally apply macOS padding without hard-coding in CSS.
- Python path resolution is multi-candidate, covering development venv, packaged `.app` Resources, system Homebrew, and `/usr/bin` fallbacks with clear error dialogs.
- Navigation policy enforced: renderer cannot navigate outside `127.0.0.1` / `localhost`.

---

## What's New (v4.0.0+ — Full Codebase)

### Backend (v4.0.0+)

- **Modular architecture** — code split into `backend/main.py` (app factory + lifespan), `backend/api/routes.py` (all route handlers), `backend/services/inference.py` (model loading + image pipeline), `backend/services/cache_service.py` (Redis-backed cache with in-memory fallback), `backend/services/benchmarks.py` (runtime performance matrices), and `backend/utils/hardware.py` (device detection).
- **`backend/config.py`** — new `Settings` class backed by `pydantic-settings`. Reads `HOST`, `PORT`, `LOG_LEVEL`, `DEBUG`, `REDIS_URL`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_SOCKET_TIMEOUT_SECONDS`, `REDIS_MAX_CONNECTIONS`, and `CACHE_TTL_SECONDS` from environment variables and a `.env` file. Falls back to a lightweight shim when `pydantic-settings` is absent. Exported as a cached singleton `settings`.
- **Structured JSON logging** — `JsonLogFormatter` in `backend/main.py` emits every log record as a JSON object with `timestamp`, `level`, `logger`, `message`, `module`, `function`, `line`, `process`, `thread`, and optional `exception` / `stack_info` fields. All uvicorn and FastAPI loggers are routed through this formatter.
- **System telemetry in `/health`** — the health endpoint now includes a `telemetry` block with:
  - `memory` — reads `/proc/meminfo` (Linux) or `os.sysconf` page-based totals; reports `pressure_percent`, `total_bytes`, `available_bytes`, `used_bytes`, and status (`ok` / `degraded` / `unknown`).
  - `disk` — calls `shutil.disk_usage("/")` and reports `usage_percent`, `total_bytes`, `free_bytes`, `used_bytes`.
  - Overall `status` field is `"degraded"` when either subsystem exceeds its configured threshold (90% memory pressure or 90% disk usage), otherwise `"ok"`.
- **Device detection and prioritization refined:**
  - Default device priority: `CUDA > MPS > XPU > CPU`.
  - MPS runtime checks enforce both `is_built()` and `is_available()` guards.
  - XPU detected via `torch.xpu` with `device_count()` enumeration.
- **Startup lifecycle** — warm-up on best available accelerator; falls back to CPU if accelerator pre-load fails; both paths log clearly.
- **ONNX Runtime execution layer** — `backend/depth_models.py` now includes `ONNXExecutionEngine`, static `.onnx` path resolution (`DEPTHLENS_ONNX_DIR` override supported), provider selection for CUDA, CoreML on macOS, and CPU, plus legacy `DepthEstimator(prefer_onnx=True)` compatibility.
- **Static ONNX export workflow** — `backend/scripts/export_onnx.py` exports one model or all supported MiDaS models to fixed `[1, 3, 384, 384]` ONNX graphs, validates them with `onnx.checker`, and stores them in `backend/onnx_weights/` by default.
- **Inference engine preference** — `backend/services/inference.py` prefers ONNX Runtime when a matching exported graph exists and automatically falls back to PyTorch Hub inference when weights are missing. `loaded_model_keys()` reports ONNX engines with an `onnx:` prefix.
- **Benchmarking service** — `backend/services/benchmarks.py` compares PyTorch and ONNX Runtime on synthetic frames, reporting average/min/max latency, throughput, memory before/after, speedup, faster engine, resolved device, and ONNX weight availability.
- **Redis cache layer** — `backend/services/cache_service.py` stores serialized inference payloads in Redis with configurable TTL, connection pooling, socket timeouts, key prefixing, SCAN-based clearing, live metrics, and a resilient in-memory fallback with temporary Redis backoff after failures.
- **`/health` endpoint** now returns `acceleration_checks` with per-backend `available` + `operational` probe results (actual tensor multiply executed on device), the new `telemetry` block, cache size, and `cache_metrics`.
- **`/devices` endpoint** exposes `compute_classes` per device (`["gpu"]`, `["cpu"]`, etc.) for frontend filter rendering.
- Request safety enforced: per-file size cap (**20 MB**), image dimension cap (**2048 px** max side, auto-resized), batch upper limit (**10 images**), standardised JSON error responses.
- `backend/app.py` is a backward-compatible ASGI shim — existing `uvicorn app:app` launch flows from packaged Electron still work.
- `backend/depth_models.py` retained as a legacy `DepthEstimator` wrapper for callers using the old API directly, with ONNX-first prediction when exported weights exist.
- **CI pipeline** (`black`, `ruff`, `mypy`, `pytest`) fully integrated with GitHub Actions on `main`/`master` push and PRs.
- **mypy** configured with `strict = True`, `ignore_missing_imports = True`; tests excluded from type-checking via `mypy.ini`.
- `pyproject.toml` defines shared `line-length = 100` for both Black and Ruff; Ruff lints `E`, `F`, `W`, `I` rule sets and pytest uses repository-root `pythonpath` for stable test discovery.

### Frontend (v5.0 — script.js + style.css + welcome-anim.js)

- **Welcome screen v5.0** — liquid-metal canvas logo animation: blobs spawn from ring → fall toward centre → merge → morph into particle cloud → crystallise into final gradient text. 60 fps, theme-aware, reduced-motion safe.
- **Vector background canvas** on the welcome screen — grid lines (fine + bold), diagonal accents, circuit connector segments, and intersection dots; re-drawn on theme change (`depthlens-theme-changed` custom event) and window resize. Colors adapt clearly for both dark and light modes.
- **Theme system** — `dark` / `light` toggle persisted to `localStorage`; applied before first paint to eliminate flash. `applyTheme()` dispatches `depthlens-theme-changed` for `welcome-anim.js` to update the background canvas and logo colors immediately. Toggle button animates with spring physics (`cubic-bezier(0.34,1.56,0.64,1)`). A spinning multi-colour conic-gradient ring appears on hover.
- **Landing → workspace transition** — theme toggle button physically migrates from landing corner to header slot via `getBoundingClientRect` delta animation (`migrateThemeToggle()`). The button uses a `.visible` class for its final opacity/scale reveal, and the landing placeholder uses `.is-migrating` to disappear cleanly during flight.
- **Get Started button (v5.0)** — premium glass/glow button always visible at rest; spinning conic-gradient ring border at 0.78 opacity even without hover; hover triggers spring bounce (`translateY(-5px) scale(1.065)`) and full ring opacity; correct light-mode colour variants included.
- **Electron integration** — `window.electronAPI` detected at runtime; backend URL fetched via `ipcRenderer.invoke("get-backend-url")`; `macos` class applied to body for traffic-light padding; native file save/open dialogs available. `preload.js` exposes `platform` and `arch` alongside existing handles.
- **Device selector** — dynamically rendered from `/health` + `/devices`; shows filter toggle (All / CPU / GPU / NPU) when multiple classes detected; `Auto` option label describes the actual preferred device.
- **`/health` acceleration checks** surfaced in the status bar and as a warning toast if any accelerator backend is non-operational.
- **Upload flow** — drag-and-drop with animated overlay, thumbnail preview, per-item status badges, cancel button, ETA countdown, and item/total counter.
- **Batch progress** — adaptive ETA using exponential-weighted moving average of past inference times per model+device combination.
- **Lightbox v5** — overlay blend slider (original ↔ depth), accordion metric groups (6 sections, 25+ metrics), colour-coded values (good/warn/bad/na), download depth map + grayscale buttons.
- **Compare panel** — runs all three models sequentially against one image; per-model progress with ETA; 8-metric comparison bar chart (Chart.js); summary grid of best/runner-up per metric; chart metric selector.
- **Session dashboard** — 8 counters (processed, avg/min/max latency, cached, errors, throughput, total inference time) + Chart.js latency history line chart.
- **Persistent preferences** — model, colormap, and device saved to `localStorage` and restored on load.

### Electron Shell (v4.0.0)

- `main.js` — multi-candidate Python path resolution (dev venv → packaged Resources → Homebrew → system fallbacks); graceful `SIGTERM` → `SIGKILL` shutdown with 3 s timeout; log-based + health-poll backend readiness detection; hard 20 s timeout fallback opens the window anyway with offline status.
- `preload.js` — exposes `getBackendUrl`, `getAppVersion`, `getPlatform`, `showSaveDialog`, `showOpenDialog`, `platform`, `arch` via `contextBridge`.
- `package.json` — `electron-builder` config bundles `backend/`, `venv/`, and `frontend/` into `extraResources`; excludes `__pycache__` and `.pyc` files; macOS target: `dmg` / `arm64`.

---

## Project Structure

```text
DepthLensPro/
│
├── backend/                        # FastAPI + depth inference engine
│   ├── app.py                      # Backward-compatible ASGI entrypoint (shim)
│   ├── main.py                     # FastAPI app factory, JSON logging, CORS, lifespan hooks
│   ├── config.py                   # pydantic-settings Settings (runtime + Redis/cache env)
│   ├── depth_models.py             # Legacy DepthEstimator wrapper + ONNX Runtime engine
│   ├── requirements.txt            # Python dependencies (Redis, ONNX, ONNX Runtime included)
│   ├── scripts/
│   │   └── export_onnx.py          # Export MiDaS Torch Hub weights to static ONNX graphs
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # All API route handlers (incl. memory + disk telemetry)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── benchmarks.py           # PyTorch vs ONNX Runtime benchmark service
│   │   ├── cache_service.py        # Redis-backed inference cache + memory fallback
│   │   └── inference.py            # Model loading, ONNX/PyTorch image pipeline, metrics
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── hardware.py             # Device detection, resolve, acceleration checks
│   │
│   └── tests/
│       ├── conftest.py             # OpenCV stub for CI environments
│       ├── test_hardware.py        # Device fallback + selection unit tests
│       └── test_routes.py          # Route-level integration tests
│
├── frontend/                       # Web UI
│   ├── index.html                  # Multi-panel UI (Welcome / Workspace / Compare / About)
│   ├── style.css                   # Cyber-neon design system v5.0 (dark + light mode)
│   ├── script.js                   # Frontend logic v5.0 (state, API, charts, metrics, theme migrate)
│   └── welcome-anim.js             # Liquid-metal logo canvas animation v5.0
│
├── electron-app/
│   ├── main.js                     # Electron main process + backend lifecycle
│   ├── preload.js                  # Secure contextBridge API surface (platform + arch exposed)
│   ├── package.json                # Electron + electron-builder config
│   ├── package-lock.json
│   ├── entitlements.mac.plist      # macOS sandbox entitlements
│   └── src/
│       └── splash.html             # Loading screen during backend startup
│
├── .github/
│   ├── CODEOWNERS                  # All files owned by @AyushmanRaha
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: black, ruff, mypy, pytest
│
├── Dockerfile                      # Multi-stage Docker image (builder + runner)
├── docker-compose.yml              # Backend + Redis compose stack with healthchecks + resource limits
├── mypy.ini                        # Mypy strict config (tests excluded)
├── pyproject.toml                  # Black + Ruff shared config (line-length 100)
├── .gitignore
├── .dockerignore
├── CONTRIBUTING.md
├── SECURITY.md
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Configuration** | pydantic-settings (`backend/config.py`) |
| **Logging** | Structured JSON via `JsonLogFormatter` |
| **AI Models** | MiDaS (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`) via PyTorch Hub + exported ONNX graphs |
| **Inference Runtime** | ONNX Runtime first when static weights exist, PyTorch fallback otherwise |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript (ES2022) |
| **Charts** | Chart.js 4.4 |
| **Desktop Shell** | Electron v42 |
| **Build Tool** | electron-builder v26 |
| **Core Libraries** | PyTorch, OpenCV, NumPy, ONNX, ONNX Runtime |
| **Transport / Data** | Multipart uploads + JSON API responses + base64 PNG payloads |
| **Cache** | Redis with TTL + in-memory fallback |
| **Containerisation** | Docker (multi-stage), Docker Compose |
| **CI** | GitHub Actions (black, ruff, mypy, pytest) |

---

## Installation & Setup

### Desktop App (Recommended)

Download the latest release from the GitHub Releases page.

#### macOS

Mount the DMG and drag **DepthLens Pro** to Applications.

If blocked by Gatekeeper:

```bash
xattr -cr "/Applications/DepthLens Pro.app"
```

Then open normally. The inference engine starts automatically and a splash screen is shown until it is ready.

> **Note:** First-time model loading downloads weights from `intel-isl/MiDaS` via `torch.hub`, so internet access is required during the initial model bootstrap.

---

### Desktop Usage Guideline

1. Launch **DepthLens Pro** from Applications or the installed desktop shortcut.
2. Wait for the splash screen while the bundled FastAPI inference backend starts automatically.
3. Click **Get Started** to enter the workspace.
4. Upload one or more 2D images via drag-and-drop or the file browser (PNG, JPG, WEBP, BMP; max 20 MB each).
5. Select the MiDaS model, colormap, and compute device.
6. Click **Generate Depth Maps** to create colorized and grayscale depth outputs.
7. Review the generated cards, open the **Lightbox** for detailed metrics, and download outputs as needed.
8. Use the **COMPARE** panel to run all supported models against a single image and compare metrics side by side.
9. Quit the desktop app normally; the inference backend is stopped automatically.

---

### Desktop Development Mode

#### Prerequisites

- Python 3.10+
- pip
- Git
- Node.js

#### Setup

##### macOS / Linux

```bash
# Clone repository
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies used by the Electron shell
pip install -r backend/requirements.txt

# Optional: export static ONNX weights for ONNX Runtime acceleration
python backend/scripts/export_onnx.py --model MiDaS_small

# Install Electron dependencies and launch the desktop app
cd electron-app
npm install
npm start
```

##### Windows

```bash
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

python -m venv venv
venv\Scripts\activate

pip install -r backend/requirements.txt

# Optional: export static ONNX weights for ONNX Runtime acceleration
python backend/scripts/export_onnx.py --model MiDaS_small

cd electron-app
npm install
npm start
```

The Electron shell auto-detects `NODE_ENV=development` (or `app.isPackaged === false`) and opens DevTools in a detached window.

---

### Build Desktop App

```bash
cd electron-app

# macOS (arm64 DMG)
npm run build:mac

# Windows (NSIS installer)
npm run build:win

# Both
npm run build:all
```

---

## Validation & CI

Run the same checks used by the GitHub Actions pipeline locally:

```bash
black --check .
ruff check .
mypy backend/
pytest
```

All four gates must pass before opening a pull request. The pytest configuration in `pyproject.toml` sets repository-root `pythonpath` so backend tests can import `backend.*` consistently. See `CONTRIBUTING.md` for full guidelines.

---

## Workflow

### Workspace Flow

1. The animated welcome screen plays the liquid-metal logo formation (v5.0). Click **Get Started** to enter the workspace — the theme toggle migrates from the landing corner to the header via a spring-physics position animation.
2. The header status indicator shows engine connectivity, active compute device, PyTorch version, and acceleration check results.
3. Upload one or more 2D images via drag-and-drop or the file browser (PNG, JPG, WEBP, BMP; max 20 MB each).
4. Select model architecture, colormap, and compute device. Use the device filter buttons to narrow by CPU / GPU / NPU.
5. Click **Generate Depth Maps** to start the batch. A progress bar with adaptive ETA and per-item status updates is shown; cancel is available at any time.
6. Review output cards in the results gallery. Click any card to open the **Lightbox** for the blend slider, full metric accordion, and download actions.
7. Re-running identical image + model + colormap + device combinations serves results from Redis when available, or the in-memory fallback otherwise (shown as `cached` tag).
8. The Session Dashboard tracks processed count, avg/min/max latency, cache hits, errors, throughput, total inference time, and a latency history chart.

### Compare Flow

1. Switch to the **COMPARE** panel.
2. Upload a single image and choose colormap and device.
3. Click **Run All Models** — `MiDaS_small`, `DPT_Hybrid`, and `DPT_Large` run sequentially.
4. Side-by-side output cards appear as each model finishes.
5. A bar chart and summary grid compare up to 8 metrics (latency, SSIM, SILog, PSNR, gradient mean, edge density, entropy, dynamic range). Use the metric selector to switch the chart view.

---

## API Overview

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API metadata and service version |
| `GET` | `/health` | Engine status, device inventory, acceleration checks, telemetry, cache size, cache metrics, torch version |
| `GET` | `/devices` | All discovered compute targets with type, compute classes, and hardware details |
| `GET` | `/models` | Supported model registry with performance notes |
| `GET` | `/colormaps` | Supported colormap keys |
| `POST` | `/estimate` | Single-image depth estimation (form fields: `file`, `model`, `colormap`, `device`) |
| `POST` | `/batch` | Multi-image depth estimation — max 10 files (same form fields) |
| `GET` | `/benchmark` / `/api/benchmark` | PyTorch vs ONNX Runtime latency, throughput, memory, and speedup benchmark |
| `GET` | `/cache/metrics` | Live Redis/fallback cache metrics for dashboards |
| `DELETE` | `/cache` | Clear Redis-backed and in-memory inference cache entries |

### `/health` Response Shape

```json
{
  "status": "ok",
  "version": "3.1.0",
  "primary_device": "mps",
  "devices": { ... },
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
    "mps":  { "available": true,  "operational": true  },
    "xpu":  { "available": false, "operational": false }
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

### `/benchmark` Response Shape

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

### `/estimate` Response Shape

```json
{
  "depth_map":   "<base64 PNG>",
  "grayscale":   "<base64 PNG>",
  "metrics":     { "ssim": 0.812, "silog": 7.43, "psnr": 28.5, ... },
  "latency_ms":  31.4,
  "model":       "MiDaS_small",
  "colormap":    "inferno",
  "device_used": "mps",
  "resolution":  { "width": 1024, "height": 768 },
  "filename":    "photo.jpg",
  "cached":      false
}
```

---

## Metrics Reference

Metrics are split into two categories:

**Computable without ground truth** — derived directly from the predicted depth map and the input image:

| Metric | Description |
|---|---|
| `ssim` | Structural similarity between grayscale input and depth map (0–1, higher = better) |
| `silog` | Scale-invariant log error — measures log-depth variance after removing global scale |
| `psnr` | Peak signal-to-noise ratio of the depth map relative to its own mean (dB) |
| `entropy` | Shannon entropy of the depth histogram (bits) |
| `dynamic_range` | Log₂ ratio of max/min non-zero depth — larger = more depth variation |
| `coverage` | Fraction of histogram bins with ≥ 1% of peak count |
| `gradient_mean` | Mean Sobel gradient magnitude — higher = more depth edges |
| `gradient_std` | Variation in gradient strength |
| `edge_density` | Fraction of pixels with gradient > mean + std |
| `mae`, `rmse`, `log_rmse` | Error metrics relative to depth mean (self-consistency proxies) |

**Require ground-truth depth** — shown as `N/A` since only the source image is provided:

`abs_rel`, `sq_rel`, `delta_1/2/3`, `lpips`, `ordinal_error`, `surface_normal_error`

---

## Model Reference

| Model | Architecture | Approx. Speed (GPU) | Quality | Recommended Compute |
|---|---|---|---|---|
| `MiDaS_small` | EfficientNet-Lite | ~30 ms | Good | CPU or GPU |
| `DPT_Hybrid` | ViT-Hybrid | ~120 ms | Very Good | GPU recommended |
| `DPT_Large` | ViT-Large | ~400 ms | Excellent | GPU required for practical speed |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure virtual environment is activated and `pip install -r backend/requirements.txt` completed |
| Frontend shows "Engine offline" | Confirm FastAPI is running on `127.0.0.1:8000` |
| Browser CORS / network issue | Serve frontend via a local HTTP server (`python -m http.server 5500`) rather than `file://` |
| Slow inference | Use `MiDaS_small` or select `CUDA` / `MPS` / `XPU` where available |
| Port already in use | Change port with `--port XXXX` in the uvicorn command |
| First run is slow | MiDaS model weights are downloaded once and cached locally by torch.hub |
| App blocked by macOS Gatekeeper | Run: `xattr -cr "/Applications/DepthLens Pro.app"` |
| Inference engine does not start (packaged app) | Check logs at `~/Library/Logs/depthlens-pro/main.log` |
| Python not found (packaged app) | Ensure the `venv/` folder was included during build; check electron-builder `extraResources` config |
| Duplicate app icon in Applications | Delete old copy: `rm -rf "/Applications/DepthLens Pro.app"` and reinstall from DMG |
| `acceleration_ok: false` in `/health` | One or more GPU backends are unavailable or failed the tensor probe — inference still works on CPU |
| `mypy` errors in CI | Run `mypy backend/` locally; ensure new code has correct type annotations matching `strict = True` |
| `status: "degraded"` in `/health` | Memory pressure or disk usage exceeded 90% threshold — check system resources; inference continues normally |
| `.env` values not picked up | Ensure `.env` is at the repository root (same level as `pyproject.toml`); key names are case-insensitive |
| Docker container exits immediately | Verify `.env` is present (required by `docker-compose.yml`); check `docker logs <container>` for JSON-formatted startup errors |
| Redis cache unavailable | The backend automatically uses the in-memory fallback and retries Redis after a short backoff; verify `REDIS_HOST`, `REDIS_PORT`, credentials, and the Redis container healthcheck |
| ONNX Runtime unavailable or slow | Export weights with `python backend/scripts/export_onnx.py --model MiDaS_small`; otherwise inference falls back to PyTorch Hub automatically |
| `/benchmark` reports ONNX unavailable | Confirm the expected `.onnx` file exists under `backend/onnx_weights/` or set `DEPTHLENS_ONNX_DIR` to the directory containing exported graphs |

---

## Contributing

See `CONTRIBUTING.md` for full guidelines. In brief:

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Run `black --check .`, `ruff check .`, `mypy backend/`, `pytest` — all must pass.
4. Commit: `git commit -m "Describe your change"`
5. Push and open a Pull Request — PRs require approval from `@AyushmanRaha` (enforced by `CODEOWNERS`).

Preserve API response shapes and frontend compatibility unless a breaking change is discussed first.

---

## Security

See `SECURITY.md`. To report a vulnerability, do not open a public GitHub issue — contact the repository maintainers privately with a description, reproduction steps, and affected components.

---

## Future Enhancements

- Video depth estimation pipeline
- True 3D reconstruction (point clouds / meshes)
- Disk/object-store cache persistence beyond Redis
- Ground-truth metric support (Abs Rel, δ thresholds, LPIPS) via optional depth reference upload
- Mobile-responsive UX improvements

---

## License

This project is licensed under the MIT License.

---

## Authors

**Primary Author and Maintainer: Ayushman Raha**
GitHub: [https://github.com/AyushmanRaha](https://github.com/AyushmanRaha)

---

## Acknowledgements

- MiDaS Depth Estimation Models — Intel ISL
- OpenCV & PyTorch communities
- FastAPI ecosystem
- Chart.js
- Electron & electron-builder

---

## Notes

- Inference cache uses Redis when available and falls back to an in-memory TTL cache when Redis is unavailable. Use `DELETE /cache` to clear cache entries manually.
- Maximum supported image size is **20 MB** per file; maximum dimension is **2048 px** on the longest side (auto-resized above this threshold).
- GPU acceleration depends on local PyTorch backend support (`CUDA`, `MPS`, or `XPU`) and installed drivers / runtime; ONNX Runtime provider acceleration depends on the providers installed in the active environment (`CUDAExecutionProvider`, `CoreMLExecutionProvider`, or `CPUExecutionProvider`).
- The `backend/app.py` shim exists for compatibility with existing packaged Electron flows that invoke `uvicorn app:app` from inside the `backend/` directory. New integrations should prefer `uvicorn backend.app:app` from the repository root.
- All backend log output is JSON-formatted. Use `jq` or a log collector to parse it: `uvicorn backend.app:app ... 2>&1 | jq .`
- The `telemetry.memory` block falls back to `status: "unknown"` on platforms where neither `/proc/meminfo` nor `os.sysconf` is available (e.g. macOS without the relevant sysconf keys). Inference is unaffected.
