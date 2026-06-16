# DepthLens Pro

DepthLens Pro is a local Electron + FastAPI desktop application for monocular depth estimation, ONNX/PyTorch benchmarking, webcam depth previews, ground-truth evaluation, object detection diagnostics, and approximate 3D reconstruction.

## Supported native platform matrix

| Platform | Architecture | Status |
|---|---:|---|
| macOS | arm64 / Apple Silicon | Supported |
| macOS | x64 / Intel | Unsupported |
| macOS | universal | Unsupported |
| Windows | x64 | Supported |
| Windows | arm64 | Supported |
| Linux | x64 | Supported |
| Linux | arm64 / aarch64 | Supported |

macOS x64 remains intentionally unsupported. Windows x64 and Linux x64 are first-class native build targets alongside existing ARM64 targets.

## Four-step native app flow

### Step 1 — clone

```bash
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro
```

### Step 2 — setup

macOS/Linux:

```bash
./scripts/setup
```

Windows PowerShell:

```powershell
.\scripts\setup.ps1
```

NPM alias:

```bash
npm run setup:native
```

Setup detects OS/CPU architecture, rejects unsupported targets early, verifies Git/Python/Node/npm, creates or repairs `venv`, installs backend and Electron dependencies, exports/validates all required ONNX assets by default, pre-caches PyTorch fallback weights/transforms, validates resources, writes `models/manifest.json`, and writes `setup-manifest.json`.

### Step 3 — build

macOS/Linux:

```bash
./scripts/build
```

Windows PowerShell:

```powershell
.\scripts\build.ps1
```

NPM alias:

```bash
npm run build:native
```

Build reads `setup-manifest.json`, verifies the current OS/arch matches setup, verifies Python/Node resources, requires all ONNX files, runs resource checks, builds the current supported target, and verifies packaged resources. Native/release builds do not silently skip ONNX.

### Step 4 — launch

macOS/Linux:

```bash
./scripts/launch
```

Windows PowerShell:

```powershell
.\scripts\launch.ps1
```

NPM alias:

```bash
npm run launch:native
```

Launch prints the exact artifact path. If no built app exists it tells you to run the build command.

## Model asset policy

Model binaries are not committed. Setup installs or exports required assets into `models/` and records validation metadata in `models/manifest.json`.

Required ONNX files:

```text
models/onnx/midas_small.onnx
models/onnx/dpt_hybrid.onnx
models/onnx/dpt_large.onnx
```

The manifest records model id, engine, filename/path, size, SHA-256, required flag, input shape, validation status, runtime/provider validation when requested, and install/export timestamp. Verification checks file presence, non-empty size, reasonable minimum size, checksum when available, ONNX checker status, ONNX Runtime session creation, and dummy inference in deep validation mode.

ONNX Runtime is installed by default and used when validated. PyTorch fallback remains available. Development mode may use `DEPTHLENS_AUTO_EXPORT_ONNX=1` to repair missing ONNX files, but packaged/release apps are expected to contain validated assets from setup/build.

Packaged users should not run raw `backend/scripts/export_onnx.py` commands manually. If a packaged app reports missing model assets, reinstall the app or re-run the documented setup/build flow.

## Build scripts and targets

Important Electron scripts include:

```bash
cd electron-app
npm run build:mac:arm64
npm run build:win:arm64
npm run build:win:x64
npm run build:linux:arm64
npm run build:linux:x64
npm run build:all:supported
npm run verify:resources:native
npm run verify:packaged
```

`verify:resources:native` and packaged verification use `--onnx require-all --models all` for native/release flows.

## Python runtime packaging

The app still packages a target-specific `venv` resource. That venv must be created on the target OS/architecture by `./scripts/setup` or `.\scripts\setup.ps1`; do not reuse a macOS arm64 venv for Windows or Linux builds. Build verification imports packaged backend resources and checks native resource layout. This keeps the current packaging model deterministic while leaving room for a future embedded Python/PyInstaller backend runtime.

## Realtime webcam depth

The webcam pipeline now has a realtime backend route:

```http
POST /api/realtime/depth
```

It accepts binary JPEG/WebP/image bytes, skips metrics/histograms/cache by default, uses realtime-optimized settings, and returns a binary PNG depth image with latency headers. The frontend uses this route for webcam frames instead of the old base64 JSON hot path, keeps only the latest in-flight frame, drops stale frames under backpressure, tracks capture/transport/backend/render latency, and defaults low-latency mode to smaller frames and no smoothing. The existing visual design is unchanged.

Low-power/thermal controls are exposed through environment variables and capabilities:

| Variable | Purpose |
|---|---|
| `DEPTHLENS_LOW_POWER_MODE=1` | Prefer lower realtime concurrency and conservative runtime settings |
| `DEPTHLENS_REALTIME_MAX_IN_FLIGHT` | Realtime inference in-flight cap, default `1` |
| `DEPTHLENS_NORMAL_MAX_IN_FLIGHT` | Normal request concurrency cap, default `2` |
| `ORT_INTRA_OP_NUM_THREADS` | ONNX Runtime intra-op threads |
| `ORT_INTER_OP_NUM_THREADS` | ONNX Runtime inter-op threads |

Benchmarking is treated as heavy compute and should not be run concurrently with live webcam work.

## Persistent storage

Electron passes its `app.getPath("userData")` to the backend as `DEPTHLENS_USER_DATA_DIR`. Backend persistence initializes SQLite at that location with versioned tables for settings, workspace sessions, workspace items, artifacts, model assets, benchmark runs, and diagnostics history. Large artifacts are stored under app-managed directories such as `artifacts/depth-maps`, `artifacts/reconstructions`, `artifacts/thumbnails`, `cache`, and `logs`.

Renderer APIs expose safe storage operations for settings, benchmark history, model status, cache clearing, and storage path discovery. LocalStorage remains a fast UI cache/migration fallback, not the long-term source of truth. Raw user images are not persisted by default.

Typical user data locations:

| OS | Storage root |
|---|---|
| macOS | `~/Library/Application Support/DepthLens Pro` |
| Windows | `%APPDATA%\DepthLens Pro` |
| Linux | `~/.config/DepthLens Pro` |

Backend-only/dev mode defaults to `~/.depthlenspro` unless `DEPTHLENS_USER_DATA_DIR` is set.

## Ground-truth metrics

Ground truth files may be PNG, TIFF, or NPY. Predictions are aligned to GT by median scale and evaluated on valid finite positive pixels.

Implemented GT metrics include Abs Rel, Sq Rel, GT MAE, GT RMSE, GT Log RMSE, δ thresholds, GT PSNR, GT SSIM, Surface Normal Error, and Ordinal Error. LPIPS is reported as `optional_dependency_missing` unless a packaged LPIPS dependency/model is added. The inspector now prioritizes backend unavailable reasons, so uploaded GT is no longer incorrectly blamed when a metric is optional or unavailable.

## Detector diagnostics

The RGB camera/object detector route remains TorchVision-based in this phase, but diagnostics are explicit:

```http
GET /api/detect/status
```

The status reports Torch import status, TorchVision import status, default detector weights status, selected device/provider, MPS-to-CPU fallback warning where applicable, and the last load error. The UI displays actionable detector errors instead of only “detector unavailable.” A future model-manifest-backed ONNX nano detector can replace the TorchVision detector without changing the UI.

## Capability and model APIs

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/capabilities` | Platform support, devices, model readiness, detector status, storage path, realtime support, low-power status |
| `GET` | `/api/models/status` | Model manifest and ONNX/PyTorch asset status |
| `POST` | `/api/models/validate` | Deep model validation with ONNX checker/runtime session checks |
| `GET` | `/api/detect/status` | Detector diagnostics |
| `POST` | `/api/realtime/depth` | Binary low-latency webcam depth inference |
| `GET` | `/onnx/status` | ONNX path/provider diagnostics |
| `GET` | `/health` | Full backend health diagnostics |
| `GET` | `/ready` | Dependency readiness |
| `POST` | `/estimate` | Standard image depth estimation |
| `GET` | `/benchmark` / `/api/benchmark` | PyTorch vs ONNX performance measurement |

## Error taxonomy

Stable API error codes include `ESTIMATE_TIMEOUT`, `INVALID_METRICS_MODE`, `INVALID_OUTPUTS`, `INFERENCE_DEPENDENCY_UNAVAILABLE`, `MODEL_FILE_MISSING`, `MODEL_CHECKSUM_FAILED`, `ONNX_PROVIDER_UNAVAILABLE`, `DETECTOR_MODEL_MISSING`, `GT_METRIC_NOT_IMPLEMENTED`, `GT_REQUIRED`, `REALTIME_BACKPRESSURE`, `UNSUPPORTED_ARCH`, and `SETUP_INCOMPLETE`.

## Troubleshooting

### Missing ONNX files

Run setup and build again:

```bash
./scripts/setup
./scripts/build
```

Windows:

```powershell
.\scripts\setup.ps1
.\scripts\build.ps1
```

Do not manually repair packaged apps with raw backend export commands; reinstall or rebuild with the supported flow.

### Corrupt model manifest or model file

Delete the corrupt file under `models/onnx` and rerun setup. Build verification fails if required ONNX assets are missing or corrupt.

### Detector unavailable

Open `/api/detect/status` or the UI diagnostics. Re-run setup if Torch/TorchVision or detector weights are missing. On Apple Silicon, the current TorchVision detector uses CPU for reliability.

### Unsupported macOS x64

Use Apple Silicon macOS arm64, Windows x64/arm64, or Linux x64/arm64. macOS x64/universal builds are intentionally blocked.

### Setup incomplete

Run the setup command for your platform. The setup manifest is target-specific; if you switch OS/arch, rerun setup.

### High thermals

Use low-latency webcam defaults, reduce FPS/resolution, avoid detector + webcam + benchmark concurrency, and cap ONNX/PyTorch CPU threads with the documented environment variables.

## Development/test commands

```bash
venv/bin/python -m pytest backend/tests
cd electron-app && npm test
cd electron-app && npm run verify:resources:native
```

The test suite uses mocks/stubs for heavyweight ML paths where possible and should not require GPU, Redis, Docker, Playwright browsers, CUDA, MPS, XPU, or external services.

Example macOS artifact path after a successful Apple Silicon build:

```text
electron-app/dist/mac-arm64/DepthLens Pro.app
```

Packaged resource validation is implemented by `electron-app/scripts/verify-packaged-resources.js` and is invoked by the native build scripts.
