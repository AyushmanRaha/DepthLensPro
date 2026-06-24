# Production & Packaging

[← Back to README](../README.md)

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

Run setup before packaging so the Python environment, Electron dependencies, Torch cache, detector assets, and any requested ONNX files already exist. Normal build commands verify resources before packaging and fail early with remediation if required files are missing; they do not silently download or generate resources.

ONNX variants require previously generated and validated ONNX files under `models/onnx` before packaging:

```bash
scripts/build-native-macos.sh --arch arm64 --with-onnx --onnx-models all
scripts/build-native-linux.sh --arch x64 --with-onnx --onnx-models all
```

```powershell
.\scripts\build-native-windows.ps1 -Arch x64 -WithOnnx -OnnxModels all
```

Each native build script:

1. Cleans previous `dist/` output
2. Runs `verify-resources.js` to confirm all required files are present before packaging
3. Invokes `electron-builder` for the target platform and architecture
4. Runs `verify-packaged-resources.js` to confirm the packaged app contains backend, frontend, venv, and models directories

Pass `--auto-setup` (or the PowerShell equivalent) only when you explicitly want the build wrapper to invoke setup first. Otherwise, use the platform setup commands shown below and expect missing resources to stop the build with a clear “run setup first” action. ONNX builds require all selected ONNX files to already validate; missing files fail early instead of being exported during packaging.

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

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---

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
