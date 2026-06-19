<div align="center">

# DepthLens Pro

### Local-first monocular depth estimation for desktop workflows

Turn ordinary 2D images into depth maps, compare neural depth models, benchmark optional ONNX acceleration, evaluate against ground truth, and export approximate 3D point clouds — all from a desktop app running on your own machine.

<br/>

[![Desktop App](https://img.shields.io/badge/Desktop_App-v1.0.0-111827?style=for-the-badge)](electron-app/package.json)
[![API](https://img.shields.io/badge/API-v1.0.0-2563eb?style=for-the-badge)](backend/api/live.py)
[![Electron](https://img.shields.io/badge/Electron-42.3.0-47848f?style=for-the-badge&logo=electron&logoColor=white)](electron-app/package.json)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688?style=for-the-badge&logo=fastapi&logoColor=white)](backend/requirements.txt)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab?style=for-the-badge&logo=python&logoColor=white)](scripts/doctor.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](backend/requirements.txt)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.24.3-005ced?style=for-the-badge)](backend/requirements.txt)
[![Local First](https://img.shields.io/badge/Privacy-Local_First-16a34a?style=for-the-badge)](#security--privacy)
[![License: MIT](https://img.shields.io/badge/License-MIT-facc15?style=for-the-badge)](LICENSE)

<br/>

**No cloud uploads. No API keys. No subscription.**  
Images are processed through a local Electron + FastAPI + PyTorch/ONNX pipeline.

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="docs/installation.md"><strong>Installation</strong></a> ·
  <a href="docs/api.md"><strong>API</strong></a> ·
  <a href="docs/debugging.md"><strong>Troubleshooting</strong></a> ·
  <a href="#security--privacy"><strong>Security</strong></a> ·
  <a href="#documentation"><strong>Docs</strong></a>
</p>

</div>

---

## Demo

### Quick Preview

<video src="https://github.com/user-attachments/assets/6baef599-df4c-4d48-9fa5-7cbe43a08449" controls muted playsinline width="100%">
  Your browser does not support the video tag.
</video>

### Full Demo

[Watch the full-quality 3:45 demo video](https://github.com/AyushmanRaha/DepthLensPro/releases/download/v1.0-demo/demo.mp4)

<p align="center">
  <img src="docs/screenshots/workspace-generate-depth-maps.png" alt="Workspace — Generate Depth Maps" width="900"><br/>
  <img src="docs/screenshots/compare-run-all-models.png" alt="Compare — Run All Models on One Image" width="900"><br/>
  <img src="docs/screenshots/3d-reconstruction.png" alt="3D Reconstruction" width="900">
</p>

See the full screenshot tour in [`docs/user-guide.md`](docs/user-guide.md).

---

## What it does

**For everyone:** Upload a regular image and DepthLens Pro predicts a depth map showing which parts of the scene are nearer or farther away.

**For engineers:** Electron desktop UI + local FastAPI backend + PyTorch/ONNX inference + metrics + benchmarking + point-cloud export.

> **Important limitation:** DepthLens Pro predicts **relative depth**, not survey-grade real-world metric distance. It is useful for visual depth understanding, approximate geometry, model comparison, and local experimentation.

---

## Engineering Highlights

DepthLens Pro is not just a model demo; it productizes pretrained monocular-depth models inside a local desktop workflow.

| Area | Portfolio signal |
|---|---|
| Secure desktop shell | Electron context isolation, sandboxing, disabled Node integration in the renderer, and a narrow preload bridge. |
| Local inference service | FastAPI service on `127.0.0.1` with health, readiness, inference, diagnostics, and observability routes. |
| ML orchestration | PyTorch MiDaS/DPT model dispatch with optional ONNX Runtime path and model/device/colormap selection. |
| Resilient caching | Redis when configured, with in-memory fallback so local inference remains usable without external services. |
| Observability | Local Prometheus metrics and structured runtime diagnostics without hosted telemetry. |
| Evaluation | Ground-truth evaluation, benchmark metrics, and reproducible experiment exports. |
| Geometry export | Approximate PLY/OBJ point-cloud reconstruction from predicted depth. |
| Delivery | Native packaging flow plus Docker backend workflow. |
| CI | Formatting, linting, type checking, backend tests, Electron contract tests, Docker build, and aggregate status gate. |

---

## Current Verified Tech Stack

| Component | Version / support | Verified from | Role |
|---|---:|---|---|
| Desktop app | `v1.0.0` | `electron-app/package.json` | Native desktop package metadata |
| API | `v1.0.0` | `backend/api/live.py` | Local service version returned by `/` and `/live` |
| Electron | `^42.3.0` | `electron-app/package.json` | Desktop shell and packaging runtime |
| FastAPI | `0.135.3` | `backend/requirements.txt` | Local HTTP API framework |
| Python | `3.10–3.12` | `scripts/doctor.py` | Backend runtime and ML environment |
| PyTorch | `2.11.0` | `backend/requirements.txt` | MiDaS/DPT execution through Torch Hub |
| ONNX Runtime | `1.24.3` | `backend/requirements.txt` | Optional accelerated inference and benchmarks |
| OpenCV | `4.13.0.92` | `backend/requirements.txt` | Image decode, resizing, colorization, GT alignment |
| NumPy | `2.4.4` | `backend/requirements.txt` | Array processing and metric computation |
| Pillow | `12.2.0` | `backend/requirements.txt` | Image I/O |

---

## What I built vs what I used

| Built in this repository | Used / integrated honestly |
|---|---|
| Productized local inference system, FastAPI routes, request validation, response shaping, diagnostics, caching, benchmark flow, GT evaluation, reproducible experiments, export pipeline, Electron workflow, secure preload boundary, packaging checks, and documentation. | Pretrained MiDaS/DPT model architectures and weights, PyTorch, ONNX Runtime, FastAPI, Electron, OpenCV, NumPy, Pillow, Chart.js, Redis, Docker, and standard depth-metric formulas. |

DepthLens Pro does **not** claim that the MiDaS/DPT model architecture or pretrained weights were built from scratch. The engineering work is in turning those pretrained models into a secure, local-first, observable desktop product.

---

## Feature Overview

| Feature | What it enables | Screenshot / detail |
|---|---|---|
| Workspace depth generation | Upload an image, choose model/device/colormap/output options, generate depth. | [`docs/screenshots/workspace-generate-depth-maps.png`](docs/screenshots/workspace-generate-depth-maps.png) |
| Ground-truth mode | Upload GT depth and inspect benchmark-style metrics. | [`docs/screenshots/ground-truth-mode.png`](docs/screenshots/ground-truth-mode.png) |
| Webcam live depth | Stream local camera frames through live depth estimation. | [`docs/screenshots/webcam-live-depth-streaming.png`](docs/screenshots/webcam-live-depth-streaming.png) |
| Compare models | Run multiple MiDaS/DPT choices on one image for side-by-side comparison. | [`docs/screenshots/compare-run-all-models.png`](docs/screenshots/compare-run-all-models.png) |
| Performance / ONNX benchmark | Compare PyTorch and optional ONNX Runtime behavior. | [`docs/screenshots/performance-pytorch-vs-onnx.png`](docs/screenshots/performance-pytorch-vs-onnx.png) |
| Experiments | Save reproducible validation runs and exports. | [`docs/screenshots/experiments-validation-runs.png`](docs/screenshots/experiments-validation-runs.png) |
| 3D reconstruction | Export approximate PLY/OBJ point clouds from predicted depth. | [`docs/screenshots/3d-reconstruction.png`](docs/screenshots/3d-reconstruction.png) |
| Offline Guide | In-app reference for local usage. | [`docs/screenshots/guide-offline-reference.png`](docs/screenshots/guide-offline-reference.png) |
| Local observability | Inspect health, cache, latency, and metrics locally. | [`docs/user-guide.md`](docs/user-guide.md#observability) |

---

## Architecture

<p align="center">
  <img src="docs/architecture/depthlens-system-architecture.svg" alt="DepthLens Pro local-first architecture diagram" width="900">
</p>

DepthLens Pro is split into an Electron main/preload/renderer desktop layer, a local FastAPI backend, PyTorch and optional ONNX Runtime inference paths, cache services, ground-truth evaluation, and 3D reconstruction/export. Electron owns desktop lifecycle and the secure preload bridge; FastAPI owns HTTP routes and backend diagnostics; service modules own inference, cache, metrics, and export behavior.

Read the detailed Mermaid diagram, layer-responsibility table, concurrency model, and design decisions in [`docs/architecture.md`](docs/architecture.md).

---

## Build Choice: Standard vs ONNX

> **Build recommendation:** Start with the standard build unless you specifically need ONNX acceleration experiments. Standard builds use the local PyTorch/MiDaS path and keep ONNX optional. ONNX builds can increase disk usage, memory pressure, build time, and sustained CPU/GPU load. On fanless, low-memory, older, or thermally constrained machines, ONNX builds may run hotter or behave less reliably. If unsure, use the standard build first.

Standard builds do **not** require files in `models/onnx`; ONNX builds validate bundled ONNX assets for users who specifically want acceleration experiments, benchmarking, or packaged ONNX validation.

---

## Quick Start

```bash
git clone https://github.com/AyushmanRaha/DepthLensPro.git
cd DepthLensPro
npm run setup
npm run backend:dev
npm run frontend:dev
```

Verify the local backend in another terminal:

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
```

Detailed platform setup, native builds, ONNX flows, backend-only usage, and Docker notes live in [`docs/installation.md`](docs/installation.md).

---

## Supported Platforms

| Platform | Support |
|---|---|
| macOS Apple Silicon `arm64` | Supported |
| macOS `x64` / universal | Unsupported by the current native target scripts |
| Windows `arm64` | Supported |
| Windows `x64` | Supported |
| Linux `arm64` | Supported |
| Linux `x64` | Supported |

See [`docs/installation.md`](docs/installation.md) for exact setup, build, launch, standard, and ONNX commands.

---

## API Overview

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service identity and API version |
| `GET` | `/live` | Lightweight liveness and busy/idle state |
| `GET` | `/ready` | Runtime, model-asset, and inference readiness |
| `GET` | `/health` | Full health diagnostics |
| `GET` | `/devices` | Available inference devices |
| `GET` | `/models` | Supported model IDs, labels, and metadata |
| `GET` | `/colormaps` | Supported depth colormaps |
| `GET` | `/onnx/status` | ONNX provider/model availability diagnostics |
| `GET` | `/benchmark`, `/api/benchmark` | Benchmark status/results |
| `GET` | `/cache/metrics` | Cache backend and hit/miss metrics |
| `GET` | `/metrics` | Prometheus metrics text |
| `GET` | `/api/observability`, `/observability` | Local observability snapshot |
| `DELETE` | `/cache` | Clear cache via DELETE |
| `POST` | `/cache/clear` | Clear cache via POST |
| `POST` | `/estimate` | Generate depth for one image |
| `POST` | `/batch` | Generate depth for multiple images |
| `POST` | `/api/reconstruct`, `/reconstruct` | Generate approximate point-cloud output |
| `POST` | `/api/detect`, `/detect` | RGB camera/object detection helper route |

Full request fields, response fields, and curl examples are in [`docs/api.md`](docs/api.md).

---

## Testing & CI

Local checks:

```bash
python -m black --check .
python -m ruff check .
python -m mypy backend/
python -m pytest
cd electron-app && npm test
```

GitHub Actions CI runs on pushes to all branches, pull requests targeting `main`, and manual `workflow_dispatch`. Jobs include `backend-quality`, `electron-contract`, `docker-build`, and `ci-passed`.

---

## Security & Privacy

| Control | Current behavior |
|---|---|
| Local inference | Backend listens on `127.0.0.1`; normal use requires no hosted inference service. |
| No API keys | Default local workflow requires no cloud API key or subscription. |
| Electron isolation | `contextIsolation: true`, `sandbox: true`, and `nodeIntegration: false`. |
| Preload bridge | Renderer receives only a narrow API surface for backend URL, dialogs, and platform info. |
| Navigation | Navigation is allowlisted to the local app and localhost backend. |
| External links | Opened outside the app instead of navigating the renderer. |
| Backend ownership | Process shutdown checks PID metadata and command-line ownership. |
| Cache safety | JSON cache serialization; no pickle deserialization. |
| Error handling | Client-facing errors are sanitized. |
| Observability privacy | Metrics avoid raw images, filenames, local paths, and high-cardinality image identifiers. |

See [`SECURITY.md`](SECURITY.md) and [`docs/debugging.md`](docs/debugging.md) for reporting and operational details.

---

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Standard/ONNX setup, native builds, launch commands, Docker, backend-only use |
| [`docs/api.md`](docs/api.md) | Full API reference with endpoint details and curl examples |
| [`docs/architecture.md`](docs/architecture.md) | System diagram, Mermaid flow, layer responsibilities, concurrency, design decisions |
| [`docs/depth-pipeline.md`](docs/depth-pipeline.md) | Image ingestion, preprocessing, PyTorch/ONNX paths, normalization, colorization, cache keys |
| [`docs/metrics.md`](docs/metrics.md) | Prediction-only diagnostics, GT metrics, formulas, median-scale alignment |
| [`docs/debugging.md`](docs/debugging.md) | Troubleshooting, readiness model, model assets, packaged startup checks |
| [`docs/maintenance.md`](docs/maintenance.md) | Maintenance and release checklist |
| [`docs/project-structure.md`](docs/project-structure.md) | Repository map and module layout |
| [`docs/user-guide.md`](docs/user-guide.md) | Feature tour and preserved screenshots |

---

## Repository at a glance

| Path | Purpose |
|---|---|
| `frontend/` | Renderer HTML/CSS/JavaScript workflow |
| `electron-app/` | Electron main/preload code, packaging config, desktop tests |
| `backend/` | FastAPI routes, inference services, diagnostics, metrics, reconstruction |
| `scripts/` | Setup, native build, launch, and model-asset helpers |
| `docs/` | Architecture, API, user guide, screenshots, maintenance, debugging |
| `.github/workflows/ci.yml` | Formatting, linting, type checking, tests, Docker build |

Full tree: [`docs/project-structure.md`](docs/project-structure.md).

---

## License

DepthLens Pro is released under the [MIT License](LICENSE).

## Acknowledgements

DepthLens Pro integrates pretrained MiDaS/DPT depth-estimation models and the open-source Python, JavaScript, Electron, FastAPI, PyTorch, ONNX Runtime, OpenCV, NumPy, Pillow, Redis, Docker, and Chart.js ecosystems.

<br/>

<div align="center">

**Made with care by Ayushman Raha**

</div>
