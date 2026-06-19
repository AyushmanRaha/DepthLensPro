# Architecture

![DepthLens Pro architecture](architecture/depthlens-system-architecture.svg)

DepthLens Pro is a local-first desktop system with Electron for the desktop shell, FastAPI for the localhost inference API, PyTorch/ONNX Runtime for model execution, cache services for repeated runs, and evaluation/export services for GT metrics and point clouds.

```mermaid
flowchart TB
    User[User] --> Renderer[Electron Renderer]
    subgraph Electron
      Main[Main process: lifecycle, backend spawn, port checks]
      Preload[Preload bridge: narrow IPC surface]
      Renderer
    end
    subgraph API[FastAPI Backend]
      Live[/live]
      Ready[/ready]
      Routes[/estimate /batch /detect /reconstruct /benchmark]
      Observability[/metrics /observability /cache/metrics]
    end
    subgraph Runtime[Depth Runtime]
      Torch[PyTorch Torch Hub MiDaS/DPT]
      ONNX[Optional ONNX Runtime]
      GT[Ground-truth evaluator]
      Reconstruct[PLY/OBJ point-cloud builder]
    end
    subgraph Cache[Local Cache]
      Memory[In-memory LRU]
      Redis[Optional Redis TTL cache]
      Files[models/torch-cache and optional models/onnx]
    end
    Main --> API
    Preload --> Renderer
    Renderer --> Routes
    Routes --> Torch
    Routes --> ONNX
    Routes --> GT
    Routes --> Reconstruct
    Routes --> Memory
    Routes --> Redis
    ONNX --> Files
    Torch --> Files
```

## Layer responsibilities

| Layer | Key files | Responsibility |
|---|---|---|
| Electron main | `electron-app/main.js`, `electron-app/src/main/*.js` | Window lifecycle, backend process startup, port checks, PID metadata, settings, packaged paths |
| Preload bridge | `electron-app/preload.js` | Narrow context bridge for backend URL, dialogs, and platform info |
| Renderer | `frontend/index.html`, `frontend/js/*.js`, `style.css` | Tabs, uploads, charts, previews, 3D viewer, observability, guide |
| Security policy | `electron-app/src/security-policy.js` | Navigation allowlist and external-link handling |
| FastAPI app | `backend/app.py`, `backend/api/*.py` | Routes, CORS, exception handling, lifecycle, structured responses |
| Inference | `backend/services/inference.py` | Decode, preprocess, dispatch, normalize, colorize, encode, cache |
| Registry | `backend/model_registry.py` | Canonical model IDs, aliases, display metadata, ONNX path resolution |
| Cache | `backend/services/cache_service.py` | Redis integration, in-memory fallback, JSON serialization |
| Diagnostics | `backend/services/diagnostics.py`, `backend/services/onnx_diagnostics.py` | Readiness, provider discovery, model asset validation |
| Ground truth | `backend/services/ground_truth.py` | GT decoding, masking, resizing, median-scale alignment, metrics |
| Reconstruction | `backend/services/reconstruction.py` | Pinhole projection, preview downsampling, PLY/OBJ serialization |

## Concurrency model

The backend limits inference dispatch with `INFERENCE_MAX_CONCURRENCY` and protects model forward calls with per-model/device locks. This keeps local machines stable while allowing preprocessing, routing, and non-critical work to overlap where safe. ONNX sessions use the same idea with per-model/device session locks.

## Design decisions

- **Caching:** cache keys include model, colormap, device, metric/output options, image sizing, and image-content hash. Redis failures fall back to local memory.
- **Readiness:** `/live` is lightweight, while `/ready` and `/health` distinguish runtime imports, model assets, PyTorch cache, ONNX availability, and inference readiness.
- **Packaging:** build scripts verify resources before packaging and packaged resources after packaging. They fail early instead of silently downloading assets during build.
- **Local privacy:** normal inference stays on localhost and does not require hosted inference, cloud uploads, API keys, or subscriptions.
