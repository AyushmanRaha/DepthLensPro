# Installation and Builds

DepthLens Pro supports a standard PyTorch-backed desktop workflow and optional ONNX build flows. Commands below mirror the repository scripts; they do not change build behavior.

## Build choice: standard vs ONNX

**Recommended for most users:** start with the standard build. It uses the local PyTorch/MiDaS cache and keeps ONNX optional. Standard builds do not require files in `models/onnx`.

**Use ONNX builds when:** you specifically want acceleration experiments, benchmark comparisons, or bundled ONNX model validation. ONNX builds require ONNX files under `models/onnx` and can increase disk usage, memory pressure, build time, and sustained CPU/GPU load. On fanless, low-memory, older, or thermally constrained machines, ONNX builds may run hotter, take longer, or behave less reliably.

## Fast local development

```bash
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro
npm run setup
npm run backend:dev
npm run frontend:dev
```

Verify:

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
```

## Platform setup and build matrix

| Platform | Standard setup | ONNX setup | Standard build | ONNX build | Launch |
|---|---|---|---|---|---|
| macOS arm64 | `npm run setup:mac` | `npm run setup:mac:onnx` | `npm run build:mac:arm64` | `npm run build:mac:arm64:onnx` | `npm run launch:mac` |
| Windows arm64 | `npm run setup:win` | `npm run setup:win:onnx` | `npm run build:win:arm64` | `npm run build:win:arm64:onnx` | `npm run launch:win` |
| Windows x64 | `npm run setup:win` | `npm run setup:win:onnx` | `npm run build:win:x64` | `npm run build:win:x64:onnx` | `npm run launch:win` |
| Linux arm64 | `npm run setup:linux` | `npm run setup:linux:onnx` | `npm run build:linux:arm64` | `npm run build:linux:arm64:onnx` | `npm run launch:linux` |
| Linux x64 | `npm run setup:linux` | `npm run setup:linux:onnx` | `npm run build:linux:x64` | `npm run build:linux:x64:onnx` | `npm run launch:linux` |

macOS x64 and universal builds are intentionally unsupported by the current Electron scripts.

## Backend only

```bash
npm run setup
npm run backend:dev
curl http://127.0.0.1:8765/health
```

## Docker backend workflow

```bash
docker compose up --build
```

Docker is optional for normal desktop use.

## Resource verification

```bash
npm run verify:resources
npm run verify:onnx
npm run verify:onnx:required
```

Standard native builds require the PyTorch MiDaS Torch Hub cache. ONNX builds additionally require non-empty ONNX files in `models/onnx` and fail early with remediation text if those files are missing.
