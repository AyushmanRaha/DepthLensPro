# DepthLensPro

AI-powered monocular depth estimation app with a FastAPI backend and browser frontend.

## Refactor goals achieved

- **Backward-compatible runtime**: still starts with `uvicorn backend.app:app` and still serves frontend with `python3 -m http.server`.  
- **Modular backend architecture**: split into focused modules for config, device detection, model loading, metrics, image ops, service layer, and API routes.  
- **Mac terminal + Chrome workflow preserved**: run backend and frontend from terminal; open Chrome at `http://localhost:5500`.

---

## New repo structure

```text
DepthLensPro/
├── backend/
│   ├── app.py                      # compatibility ASGI entrypoint
│   ├── requirements.txt
│   ├── depth_models.py             # legacy utility (kept for compatibility)
│   └── depthlens/
│       ├── __init__.py
│       ├── main.py                 # app factory/lifespan/middleware/handlers
│       ├── api.py                  # route handlers
│       ├── service.py              # request orchestration logic
│       ├── models.py               # MiDaS model registry + inference
│       ├── devices.py              # CPU/CUDA/MPS detection + resolution
│       ├── metrics.py              # depth quality metric suite
│       ├── image_ops.py            # decode/colorize/base64/cache key helpers
│       ├── config.py               # constants + model/colormap definitions
│       └── runtime.py              # in-memory caches/state
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── scripts/
│   ├── run_backend.sh
│   └── run_frontend.sh
├── Makefile
└── README.md
```

---

## Run on macOS (terminal + Chrome)

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) Start backend (terminal tab 1)

```bash
./scripts/run_backend.sh
# or: make backend
```

### 3) Start frontend (terminal tab 2)

```bash
./scripts/run_frontend.sh
# or: make frontend
```

### 4) Open Chrome

Go to:

- `http://localhost:5500`

Backend health check:

```bash
make health
# or curl http://127.0.0.1:8000/health
```

---

## API endpoints (unchanged behavior)

- `GET /`
- `GET /health`
- `GET /devices`
- `GET /models`
- `GET /colormaps`
- `POST /estimate`
- `POST /batch`
- `DELETE /cache`

---

## Future FAANG-level SDE-1 upgrades (recommended roadmap)

1. **Testing & quality gates**
   - Add `pytest` unit + integration tests for service, API, and model adapters.
   - Add `ruff`, `black`, and `mypy` with CI enforcement.

2. **Production architecture**
   - Introduce `Dockerfile` + `docker-compose`.
   - Add Redis for shared cache and job queue for async batch workloads.

3. **Observability**
   - Structured JSON logging, request IDs, OpenTelemetry traces, Prometheus metrics.

4. **Resilience & security**
   - Rate limiting, upload scanning, stricter CORS, payload validation, and auth for mutation endpoints.

5. **Frontend modularization**
   - Move current JS into modules (API client, state store, rendering, charts).
   - Add build tooling (Vite) + component tests.

6. **Scalability features**
   - Background task processing for heavy models.
   - Pagination/history and persistent artifact storage.

7. **Developer experience**
   - Pre-commit hooks, conventional commits, semantic versioning, and release notes automation.

---

## Quick commands

```bash
make install
make backend
make frontend
make health
```
