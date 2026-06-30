# Troubleshooting

[← Back to README](../README.md)

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

Renderer startup is dependency-ordered and failure-isolated in both packaged Electron installs and development runs. Foundational state helpers now load before chart/decorative scripts, and optional renderer initialization such as background canvas animation, first-party chart setup, Compare controls, pointer glow, guide accordion, and scroll navigation is isolated from backend startup. If a decorative or chart dependency is unavailable, the app should log a console warning while backend URL resolution, `/live`, `/ready`, `/health` diagnostics, device discovery, polling, and the Depth Engine status panel continue to initialize normally.

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

Build scripts verify repo resources before packaging and packaged resources after building, but installed stale copies can still be launched accidentally. Verify resources before packaging, then verify the packaged output directly after building:

```bash
npm run verify:resources
cd electron-app
npm run verify:packaged:mac     # darwin arm64
npm run verify:packaged:win     # win32 arm64
npm run verify:packaged:linux   # linux arm64
node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --torch-cache required --onnx optional
node scripts/verify-packaged-resources.js --platform win32 --arch x64 --mode native --torch-cache required --onnx optional
node scripts/verify-packaged-resources.js --platform linux --arch x64 --mode native --torch-cache required --onnx optional
```

---

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

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---


## Error envelopes and privacy-safe diagnostics

API errors include `detail.error_code` and `detail.message`; batch and compare item errors also include `error_detail` while preserving legacy string fields. Timeout errors are retryable and use route-specific codes. Logs and observability snapshots redact local paths, image filenames, cache tokens, and long base64-like payloads before display.


## Blank chart panels

Charts are rendered locally by `frontend/js/charts.js`. If Workspace, Performance, Benchmark, or Compare chart panels are blank, check the browser/Electron console for `DepthLens Pro:` chart diagnostics, confirm `frontend/js/charts.js` is present, run Electron resource verification, and ensure no stale `vendor/chart.umd.min.js` placeholder script is being loaded.

## Packaged ONNX runtime diagnostics

`/live` is intentionally lightweight: it reports process liveness and benchmark busy state without importing inference, model, Torch, OpenCV, NumPy, ONNX Runtime, or scanning model assets. Use deeper endpoints for runtime checks:

```bash
curl http://127.0.0.1:8765/live
curl 'http://127.0.0.1:8765/ready?depth=quick'
curl 'http://127.0.0.1:8765/onnx/status?depth=deep&force=true'
curl http://127.0.0.1:8765/api/detect/status
```

ONNX asset availability is not the same as ONNX Runtime session availability. Packaged builds may contain `.onnx` files while a preferred execution provider is unavailable or unable to create a session. When **Force ONNX Runtime** is selected, DepthLens tries the preferred provider first and then retries a generic `CPUExecutionProvider` ONNX Runtime session when available. PyTorch fallback is used only when ONNX Runtime cannot create any valid session, the file is missing or empty, or `onnxruntime` cannot be imported.

The Compare tab includes its own engine selector and forwards that choice to every model run. The 3D camera detector now exposes `/api/detect/status` and warms up the TorchVision detector before polling frames; capture remains usable if detector dependencies or weights are unavailable. Packaged builds should include detector weights under the documented Torch cache or run setup with detector prefetch enabled.


### Depth engine delayed/offline

“Depth engine delayed” means the last live probe timed out or was slow, but recent liveness/readiness or an active frontend operation still indicates the backend may be working. Do not restart solely because inference is running. “Depth engine offline” is reserved for stronger failures such as connection refusal or repeated failures with no active operation and no recent readiness.

### ONNX failures

Prefer ONNX Runtime first tries the selected provider, then CPUExecutionProvider when safe, then PyTorch fallback if allowed. Force ONNX Runtime Strict does not fall back to PyTorch; it returns structured diagnostics with the ONNX failure stage and root exception.

### DPT Large

DPT Large is opt-in in Compare. It is high quality but slow and memory-heavy, especially on CPU/MPS. A DPT Large timeout is a model-specific failure, not evidence that the backend is offline.

### Detector loaded but no labels

“Detector loaded · detecting…” means the TorchVision model is available and polling frames. “No object detected” is a successful empty detection result at the shown threshold. Capture remains available even if the detector is unavailable.

### Missing packaged resources

Run `npm --prefix electron-app run verify:resources -- --mode release --onnx require-all --models all --torch-cache required --detector-cache required` before release packaging. Release mode fails on missing backend/frontend resources, venv/Python, PyTorch model cache, detector cache, or advertised ONNX files.
