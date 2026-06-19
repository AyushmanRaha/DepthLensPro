# Debugging and Troubleshooting

## Basic checks

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/health
```

`/live` confirms the process is reachable. `/ready` separates backend liveness from runtime imports, model assets, PyTorch cache, ONNX availability, and inference readiness. `/health` provides deeper diagnostics.

## Backend offline

Start the backend with:

```bash
npm run backend:dev
```

If port `8765` is already in use, stop the old backend with:

```bash
npm run stop:backend
```

## Missing model assets

Standard builds need the PyTorch MiDaS cache under `models/torch-cache`. ONNX is optional for standard builds. ONNX builds need `.onnx` files under `models/onnx`.

```bash
npm run setup
npm run verify:resources
npm run verify:onnx
```

## Packaged startup readiness model

Packaged startup separates:

1. Electron main process starts and resolves packaged paths.
2. Backend process is spawned without shell execution.
3. Electron polls `/live` until the local service is reachable.
4. `/ready` reports runtime import readiness, model-asset readiness, PyTorch cache readiness, ONNX readiness, and inference readiness.
5. Renderer controls should remain disabled until the app has a usable backend state.

This model helps distinguish process launch failures, missing resources, optional ONNX gaps, and true inference failures.

## Setup appears stuck

Setup streams pip, npm, MiDaS, detector, ONNX export, and verification output. The last printed command usually identifies the active operation. Re-run the platform setup command if network or cache operations fail.

## ONNX benchmark unavailable

ONNX Runtime is optional. Use standard PyTorch inference unless you intentionally need ONNX acceleration experiments. For ONNX builds, run the ONNX setup script for your platform and validate files in `models/onnx`.
