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
