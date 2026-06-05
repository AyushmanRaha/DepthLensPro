# DepthLensPro
**AI-Powered 2D to 3D Depth Map Generation Desktop Application**

DepthLensPro is an intelligent native cross-platform desktop application built with Electron that converts 2D images into high-quality depth visualizations using monocular depth estimation (MiDaS family). It combines a FastAPI inference backend (**v4.0.0**) with a production-style browser workspace for batch processing, device-aware execution, model comparison, and runtime analytics.

---

## Features

- AI-powered monocular depth estimation with multiple MiDaS models (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`)
- Native cross-platform desktop application via **Electron v42** with automatic backend lifecycle management
- Liquid-metal animated welcome screen with theme-aware canvas logo animation
- Persistent dark/light **theme toggle** (migrates from landing screen to header on entry)
- Batch processing support (up to **10 images** per request)
- Broad image format compatibility (PNG, JPG/JPEG, WEBP, BMP)
- Dual output rendering (colorized depth map + grayscale depth map)
- Rich colormap support (`inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, `turbo`)
- Compute-device routing with `Auto` and explicit `CPU`, `CUDA`, `MPS`, and `XPU` selection
- Device filter toggle (All / CPU / GPU / NPU) dynamically generated from detected hardware
- Expanded MDE metric suite (SSIM, SILog, PSNR, entropy, dynamic range, edge/gradient statistics, histogram spread, and consistency proxies)
- Dedicated model comparison panel with side-by-side output cards and metric-driven charting
- Session analytics dashboard (latency trends, throughput, cache hits, error counts, total inference time)
- In-memory inference caching for repeated image + model + colormap + device combinations
- API-first backend architecture for integration, automation, and deployment workflows
- Per-request upload cancel support with ETA/progress visibility
- Persistent user preferences (model, colormap, device) via localStorage

---

## Desktop Application

- DepthLens Pro ships as a native desktop application built with **Electron v42** — a double-clickable `.app` on macOS and `.exe` on Windows.
- The FastAPI inference backend starts and stops automatically as a child process — no manual uvicorn command needed.
- The app targets **arm64 (Apple Silicon)** natively, running PyTorch MPS inference via Metal with no Rosetta translation overhead.
- A splash screen is displayed while the backend warms up. Main window appears only after the engine is ready (or a graceful timeout).
- Distributed as a **DMG** for macOS. No internet connection required after first model weight download.
- **Security:** `contextIsolation` enabled, `nodeIntegration` disabled, renderer sandboxed. Backend only accessible on `127.0.0.1`.
- macOS traffic-light buttons use `titleBarStyle: "hidden"` with `trafficLightPosition` for correct header layout.
- Python path resolution is multi-candidate, covering development venv, packaged `.app` Resources, and system fallbacks with clear error dialogs.
- Navigation policy enforced: renderer cannot navigate outside `127.0.0.1` / `localhost`.

---

## What's New (v4.0.0 — Full Codebase)

### Backend (v4.0.0)

- **Modular architecture** — code split into `backend/main.py` (app factory + lifespan), `backend/api/routes.py` (all route handlers), `backend/services/inference.py` (model loading + image pipeline), `backend/services/cache_service.py` (in-memory cache), and `backend/utils/hardware.py` (device detection).
- **Device detection and prioritization refined:**
  - Corrected compute-class semantics for Apple MPS and Intel XPU.
  - Default device priority: `CUDA > MPS > XPU > CPU`.
  - MPS runtime checks enforce both build-time (`is_built()`) and runtime (`is_available()`) availability guards.
  - XPU detected via `torch.xpu` with `device_count()` enumeration.
- **Startup lifecycle** — warm-up on best available accelerator; falls back to CPU if accelerator pre-load fails; both paths log clearly.
- **`/health` endpoint** now returns `acceleration_checks` with per-backend `available` + `operational` probe results (actual tensor multiply executed on device).
- **`/devices` endpoint** exposes `compute_classes` per device (`["gpu"]`, `["cpu"]`, etc.) for frontend filter rendering.
- Request safety enforced: per-file size cap (**20 MB**), image dimension cap (**2048 px** max side, auto-resized), batch upper limit (**10 images**), standardised JSON error responses.
- `backend/app.py` is a backward-compatible ASGI shim — existing `uvicorn app:app` launch flows from packaged Electron still work.
- `backend/depth_models.py` retained as a legacy `DepthEstimator` wrapper for callers using the old API directly.
- **CI pipeline** (`black`, `ruff`, `mypy`, `pytest`) fully integrated with GitHub Actions on `main`/`master` push and PRs.
- **mypy** configured with `strict = True`, `ignore_missing_imports = True`; tests excluded from type-checking via `mypy.ini`.
- `pyproject.toml` defines shared `line-length = 100` for both Black and Ruff; Ruff lints `E`, `F`, `W`, `I` rule sets.

### Frontend (v5.0 — script.js + style.css + welcome-anim.js)

- **Welcome screen v5.0** — liquid-metal canvas logo animation: blobs spawn from ring → fall toward centre → merge → morph into particle cloud → crystallise into final gradient text. 60 fps, theme-aware, reduced-motion safe.
- **Vector background canvas** on the welcome screen — grid lines, diagonal accents, circuit connector segments, and intersection dots; re-drawn on theme change and window resize.
- **Theme system** — `dark` / `light` toggle persisted to `localStorage`; applied before first paint to eliminate flash. Theme toggle button animates with spring physics (`cubic-bezier(0.34,1.56,0.64,1)`). A spinning multi-colour conic-gradient ring appears on hover.
- **Landing → workspace transition** — theme toggle button physically migrates from landing corner to header slot via `getBoundingClientRect` delta animation.
- **Electron integration** — `window.electronAPI` detected at runtime; backend URL fetched via `ipcRenderer.invoke("get-backend-url")`; `macos` class applied to body for traffic-light padding; native file save/open dialogs available.
- **Device selector** — dynamically rendered from `/health` + `/devices`; shows filter toggle (All / CPU / GPU / NPU) when multiple classes detected; `Auto` option label describes the actual preferred device.
- **`/health` acceleration checks** surfaced in the status bar and as a warning toast if any accelerator backend is non-operational.
- **Upload flow** — drag-and-drop with animated overlay, thumbnail preview, per-item status badges, cancel button, ETA countdown, and item/total counter.
- **Batch progress** — adaptive ETA using exponential-weighted moving average of past inference times per model+device combination.
- **Lightbox v5** — overlay blend slider (original ↔ depth), accordion metric groups (6 sections, 25+ metrics), colour-coded values (good/warn/bad/na), download depth map + grayscale buttons.
- **Compare panel** — runs all three models sequentially against one image; per-model progress with ETA; 8-metric comparison bar chart (Chart.js); summary grid of best/runner-up per metric; chart metric selector.
- **Session dashboard** — 8 counters (processed, avg/min/max latency, cached, errors, throughput, total inference time) + Chart.js latency history line chart.
- **Persistent preferences** — model, colormap, and device saved to `localStorage` and restored on load.

### Electron Shell (v4.0.0)

- `main.js` — multi-candidate Python path resolution (dev venv → packaged Resources → system fallbacks); graceful `SIGTERM` → `SIGKILL` shutdown with 3 s timeout; log-based + health-poll backend readiness detection; hard 20 s timeout fallback opens the window anyway with offline status.
- `preload.js` — exposes `getBackendUrl`, `getAppVersion`, `getPlatform`, `showSaveDialog`, `showOpenDialog`, `platform`, `arch` via `contextBridge`.
- `package.json` — `electron-builder` config bundles `backend/`, `venv/`, and `frontend/` into `extraResources`; excludes `__pycache__` and `.pyc` files; macOS target: `dmg` / `arm64`.

---

## Project Structure

```text
DepthLensPro/
│
├── backend/                        # FastAPI + depth inference engine
│   ├── app.py                      # Backward-compatible ASGI entrypoint (shim)
│   ├── main.py                     # FastAPI app factory, CORS, lifespan hooks
│   ├── depth_models.py             # Legacy DepthEstimator wrapper
│   ├── requirements.txt            # Python dependencies
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # All API route handlers
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cache_service.py        # In-memory inference cache
│   │   └── inference.py            # Model loading, image pipeline, metrics
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
│   ├── style.css                   # Cyber-neon design system (dark + light mode)
│   ├── script.js                   # Frontend logic v5.0 (state, API, charts, metrics)
│   └── welcome-anim.js             # Liquid-metal logo canvas animation v5.0
│
├── electron-app/
│   ├── main.js                     # Electron main process + backend lifecycle
│   ├── preload.js                  # Secure contextBridge API surface
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
├── mypy.ini                        # Mypy strict config (tests excluded)
├── pyproject.toml                  # Black + Ruff shared config (line-length 100)
├── .gitignore
├── CONTRIBUTING.md
├── SECURITY.md
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **AI Models** | MiDaS (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`) via PyTorch Hub |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript (ES2022) |
| **Charts** | Chart.js 4.4 |
| **Desktop Shell** | Electron v42 |
| **Build Tool** | electron-builder v26 |
| **Core Libraries** | PyTorch, OpenCV, NumPy |
| **Transport / Data** | Multipart uploads + JSON API responses + base64 PNG payloads |
| **CI** | GitHub Actions (black, ruff, mypy, pytest) |

---

## Installation & Setup

### Variant A — Desktop App (Recommended)

Download the latest release from the GitHub Releases page.

#### macOS

Mount the DMG and drag **DepthLens Pro** to Applications.

If blocked by Gatekeeper:

```bash
xattr -cr "/Applications/DepthLens Pro.app"
```

Then open normally. The inference engine starts automatically and a splash screen is shown until it is ready.

---

### Variant B — Manual Setup (Development)

#### Prerequisites

- Python 3.10+
- pip
- Git
- Node.js (only if working on the Electron shell)

> **Note:** First-time model loading downloads weights from `intel-isl/MiDaS` via `torch.hub`, so internet access is required during the initial model bootstrap.

---

#### Backend Setup

##### macOS / Linux

```bash
# Clone repository
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run backend server
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

##### Windows

```bash
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro

python -m venv venv
venv\Scripts\activate

pip install -r backend/requirements.txt

uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

> The packaged Electron app launches uvicorn from inside the `backend/` directory using `uvicorn app:app`. Both invocation styles (`backend.app:app` and `app:app`) are supported via the `backend/app.py` shim.

---

#### Running the Full Application (Manual)

**1. Start the backend:**

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

**2. Serve the frontend:**

```bash
cd frontend
python -m http.server 5500
```

**3. Open in browser:**

```
http://localhost:5500
```

**4. Verify backend health (optional):**

```bash
curl http://127.0.0.1:8000/health
```

---

#### Electron Development Mode

```bash
cd electron-app
npm install
npm start
```

The Electron shell auto-detects `NODE_ENV=development` (or `app.isPackaged === false`) and opens DevTools in a detached window.

---

#### Build Desktop App

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

All four gates must pass before opening a pull request. See `CONTRIBUTING.md` for full guidelines.

---

## Workflow

### Workspace Flow

1. The animated welcome screen plays the liquid-metal logo formation. Click **Get Started** to enter the workspace (the theme toggle migrates to the header).
2. The header status indicator shows engine connectivity and the active compute device.
3. Upload one or more 2D images via drag-and-drop or the file browser (PNG, JPG, WEBP, BMP; max 20 MB each).
4. Select model architecture, colormap, and compute device. Use the device filter buttons to narrow by CPU / GPU / NPU.
5. Click **Generate Depth Maps** to start the batch. A progress bar with ETA and per-item status updates is shown; cancel is available at any time.
6. Review output cards in the results gallery. Click any card to open the **Lightbox** for the blend slider, full metric accordion, and download actions.
7. Re-running identical image + model + colormap + device combinations serves results from the in-memory cache (shown as `cached` tag).
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
| `GET` | `/health` | Engine status, device inventory, acceleration checks, cache size, torch version |
| `GET` | `/devices` | All discovered compute targets with type, compute classes, and hardware details |
| `GET` | `/models` | Supported model registry with performance notes |
| `GET` | `/colormaps` | Supported colormap keys |
| `POST` | `/estimate` | Single-image depth estimation (form fields: `file`, `model`, `colormap`, `device`) |
| `POST` | `/batch` | Multi-image depth estimation — max 10 files (same form fields) |
| `DELETE` | `/cache` | Clear the in-memory inference cache |

### `/health` Response Shape (abbreviated)

```json
{
  "status": "ok",
  "version": "4.0.0",
  "primary_device": "mps",
  "devices": { ... },
  "loaded_models": ["MiDaS_small:mps"],
  "cache_entries": 3,
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
  "system": {
    "os": "macOS-15.x-arm64",
    "machine": "arm64",
    "cpu": "Apple M3 Pro",
    "accelerators": ["GPU · Apple M3 Pro (Metal)"]
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
- Persistent cache layer (Redis / disk)
- Dockerized deployment workflow
- Mobile-responsive UX improvements
- Ground-truth metric support (Abs Rel, δ thresholds, LPIPS) via optional depth reference upload

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

- Inference cache is in-memory and resets when the backend restarts. Use `DELETE /cache` to clear it manually, or via the session dashboard if exposed in a future UI update.
- Maximum supported image size is **20 MB** per file; maximum dimension is **2048 px** on the longest side (auto-resized above this threshold).
- GPU acceleration depends on local PyTorch backend support (`CUDA`, `MPS`, or `XPU`) and installed drivers / runtime.
- The `backend/app.py` shim exists for compatibility with existing packaged Electron flows that invoke `uvicorn app:app` from inside the `backend/` directory. New integrations should prefer `uvicorn backend.app:app` from the repository root.
