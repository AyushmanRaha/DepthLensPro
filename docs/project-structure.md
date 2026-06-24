# Project Structure

[← Back to README](../README.md)

<details>
<summary>Show full repository tree</summary>

```text
DepthLensPro/
├── backend/
│   ├── api/
│   │   ├── live.py                  # / and /live routes (no heavy imports)
│   │   ├── routes.py                # Thin route orchestration — estimate, batch, health, benchmark, cache, reconstruct
│   │   ├── validation.py            # Request field, upload, model, colormap, metrics/output validation helpers
│   │   ├── errors.py                # Public HTTP error payload mapping helpers
│   │   ├── device_state.py          # Cached device discovery, accelerator checks, readiness diagnostics
│   │   └── system_telemetry.py      # Memory/disk telemetry helpers used by /health
│   ├── core/
│   │   └── paths.py                 # Central repo/model/Torch-cache/ONNX path policy and env overrides
│   ├── services/
│   │   ├── benchmarks.py            # PyTorch vs ONNX benchmarking, auto-export, busy flag
│   │   ├── cache_service.py         # Redis + in-memory LRU, versioned JSON serialisation
│   │   ├── diagnostics.py           # Module importability, ONNX weight inventory
│   │   ├── ground_truth.py          # GT decode, nearest-neighbour resize, median-scale align, metrics
│   │   ├── inference.py             # Public façade/orchestrator for image-to-depth responses
│   │   ├── inference_types.py       # Shared typed payload metadata for inference internals
│   │   ├── image_io.py              # Decode, depth normalization, colorization, PNG base64 encoding
│   │   ├── inference_cache.py       # Stable inference/depth cache keys and in-memory depth cache
│   │   ├── model_assets.py          # Runtime model asset readiness and remediation helpers
│   │   ├── model_runtime.py         # PyTorch MiDaS model/transform caches, locks, inference
│   │   ├── object_detection.py      # TorchVision RGB detection weight checks and inference
│   │   ├── observability.py         # Local-only counters, timings, and bounded histories
│   │   ├── onnx_inference.py        # ONNX engine cache, locks, inference, PyTorch fallback metadata
│   │   ├── metrics.py               # Metrics mode parsing, fast/full metrics, grouped payloads
│   │   ├── onnx_diagnostics.py      # ONNX session creation, provider selection, checker validation
│   │   └── reconstruction.py        # Pinhole projection, PLY/OBJ serialisation, preview downsampling
│   ├── scripts/
│   │   └── export_onnx.py           # ONNX export (torch.onnx + dynamo strategies), quarantine, validation
│   ├── tests/                       # Pytest suite — no GPU, Redis, or real model weights required
│   ├── utils/
│   │   └── hardware.py              # Device discovery, ONNX provider mapping, acceleration probe
│   ├── app.py                       # ASGI compatibility entry point (adds repo root to sys.path)
│   ├── constants.py                 # Shared lightweight literals for upload, modes, ONNX IDs, resource modes
│   ├── config.py                    # Pydantic settings with dotenv fallback
│   ├── depth_models.py              # ONNXExecutionEngine and DepthEstimator compatibility imports
│   ├── main.py                      # FastAPI app factory, CORS, JSON logging, lifespan
│   ├── model_metadata.py            # Lightweight aliases for backwards-compatible imports
│   ├── model_registry.py            # Canonical model specs, alias normalisation, ONNX path resolution
│   └── requirements.txt
│
├── electron-app/
│   ├── assets/                      # App icons for all platforms
│   ├── scripts/                     # Packaging, verification, lifecycle helpers
│   ├── src/
│   │   ├── main/                    # Focused main-process modules for desktop lifecycle and backend control
│   │   │   ├── backend-http.js       # /live JSON probes and DepthLens backend detection
│   │   │   ├── backend-lifecycle.js  # Backend startup, missing-resource errors, stale cleanup, safe shutdown
│   │   │   ├── backend-pid-store.js  # Private PID and backend metadata files
│   │   │   ├── paths.js              # App-root/resource/log path helpers
│   │   │   ├── ports.js              # Port availability and fallback discovery
│   │   │   ├── python-resolver.js    # Packaged/dev Python candidate selection
│   │   │   ├── settings-store.js     # Persisted settings schema, sanitization, corruption backups
│   │   │   └── windows.js            # Splash and main BrowserWindow options
│   │   ├── backend-process-policy.js # Backend ownership checks (command-line + PID metadata)
│   │   ├── platform-targets.js       # Central supported OS/architecture map, platform targets, settings bridge tests
│   │   └── security-policy.js        # Navigation allowlist, external URL classification
│   ├── main.js                      # Small Electron composition file — app lifecycle, IPC registration, module wiring
│   ├── preload.js                   # Narrow contextBridge surface
│   └── package.json
│
├── frontend/
│   ├── index.html                   # App shell and all workspace panels (Workspace, Webcam, Compare, Performance, Experiments, 3D, Guide)
│   ├── js/                          # Ordered browser scripts; ordered browser modules loaded directly by index.html
│   │   ├── state.js                 # Shared renderer state containers and defaults
│   │   ├── settings.js              # Theme, settings, localStorage, Electron settings bridge
│   │   ├── api-client.js            # Backend URL resolution, fetch wrappers, health/device/status calls
│   │   ├── dom.js                   # DOM lookup map and safe element helpers
│   │   ├── uploads.js               # File queue, validation, drag/drop
│   │   ├── inference-ui.js          # Estimate/batch progress and result rendering
│   │   ├── webcam.js                # Real-time camera inference lifecycle and cleanup
│   │   ├── compare.js               # Model comparison workspace
│   │   ├── benchmark.js             # PyTorch/ONNX benchmark panel
│   │   ├── experiments.js           # Experiment run/export workspace
│   │   ├── reconstruction.js        # 3D reconstruction and point-cloud preview
│   │   ├── charts.js                # Chart.js setup and theme handling
│   │   ├── notifications.js         # Settings panels, persistence hooks, toasts/navigation helpers
│   │   ├── observability.js         # Gallery/lightbox and observability dashboard
│   │   ├── performance.js           # Metrics dashboard refresh helpers
│   │   └── compat.js                # Final initialization, cleanup, utility compatibility
│   ├── script.js                    # Compatibility breadcrumb; index.html loads frontend/js directly
│   ├── style.css                    # Full design system (dark + light theme, CSS variables, animations)
│   └── welcome-anim.js              # Depth Field Calibration canvas animation (self-contained)
│
├── models/
│   └── onnx/                        # Generated ONNX files (not committed)
│
├── scripts/
│   ├── doctor.py                    # Cross-platform setup and environment verification
│   ├── diagnose_backend.py          # Port/process/endpoint diagnostics
│   ├── prefetch-detector-weights.py # Optional detector-weight cache prefetch
│   ├── prefetch-midas-assets.py     # MiDaS Torch Hub/checkpoint cache prefetch
│   ├── setup_state.py               # Shared setup report/status helpers
│   ├── setup-macos.sh
│   ├── setup-linux.sh
│   ├── setup-windows.ps1
│   ├── build-native-macos.sh
│   ├── build-native-linux.sh
│   └── build-native-windows.ps1
│
├── docs/                           # Markdown guides and reference material
│   ├── api-reference.md
│   ├── configuration.md
│   ├── debugging.md
│   ├── depth-metrics.md
│   ├── ground-truth-evaluation.md
│   ├── how-depth-estimation-works.md
│   ├── maintenance.md
│   ├── models-colormaps-metrics.md
│   ├── production-packaging.md
│   ├── project-structure.md
│   ├── refactor-test-matrix.md
│   ├── security.md
│   ├── setup-and-build.md
│   ├── system-architecture.md
│   ├── system-design-decisions.md
│   ├── terminal-only-development.md
│   ├── testing-and-ci.md
│   └── troubleshooting.md
│
├── .github/
│   └── workflows/ci.yml             # GitHub Actions — lint, type-check, pytest, Electron tests, resource dry-run
├── Dockerfile                       # Two-stage Python 3.12 slim build
├── docker-compose.yml
├── package.json                     # Root npm scripts
├── pyproject.toml                   # Black, Ruff, pytest configuration
├── mypy.ini
├── LICENSE
└── README.md
```

</details>

### Backend route organization

The inference service `backend/services/inference.py` preserves the existing public imports and response payloads delegates image I/O, cache keys, PyTorch runtime, ONNX fallback handling, and metrics to focused sibling modules.

The FastAPI route layer stays thin: public route declarations and high-level orchestration remain in `backend/api/routes.py`, while reusable validation, error mapping, device/readiness cache, and health telemetry helpers live in the neighboring `backend/api/` modules listed above. Public API routes, request fields, status codes, and response shapes are documented in the API Reference.

### Centralized path, platform, and model policy

DepthLens Pro keeps low-risk shared literals and path resolution in small central modules so setup, backend runtime, and packaging checks can stay in sync without changing public commands or API payloads. Backend upload/mode/model constants live in `backend/constants.py`, filesystem roots and ONNX environment override priority live in `backend/core/paths.py`, and Electron platform support remains centralized in `electron-app/src/platform-targets.js`. Standard builds still treat ONNX files as optional, while ONNX builds continue to require all supported ONNX model files.

### Reliability/performance hardening

Runtime hardening reduces avoidable work: Estimate cache hits now reuse normalized output metadata instead of reparsing it for telemetry, local observability hooks are failure-safe around inference paths, ONNX provider/device metadata is normalized before selection without changing provider priority, and Electron startup diagnostics keep a bounded backend-output tail with setup-remediation context.

---

Maintainer documentation lives in [`docs/maintenance.md`](maintenance.md) and [`docs/debugging.md`](debugging.md).
