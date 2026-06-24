# Configuration

[← Back to README](../README.md)

Settings are read from environment variables, with optional fallback to a `.env` file in the repository root. `pydantic-settings` is used when available; a lightweight dotenv parser handles the case where dependencies are not yet installed.

### Safe Local `.env`

```env
HOST=127.0.0.1
PORT=8765
LOG_LEVEL=INFO
DEBUG=false

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL_SECONDS=3600
CACHE_MAX_ENTRIES=256

DEPTHLENS_PRELOAD_MODEL=false
DEPTHLENS_WARMUP_MODEL=MiDaS_small
DEPTHLENS_WARMUP_DEVICE=auto
DEPTHLENS_MAX_DIM=1536
DEPTHLENS_DEFAULT_METRICS=fast
DEPTHLENS_DEFAULT_OUTPUTS=color
DEPTHLENS_OBSERVABILITY_ENABLED=true
DEPTHLENS_PROMETHEUS_ENABLED=true
DEPTHLENS_TELEMETRY_MAX_EVENTS=200
DEPTHLENS_TRACE_HISTORY_LIMIT=200
DEPTHLENS_CRASH_HISTORY_LIMIT=100
DEPTHLENS_BENCHMARK_HISTORY_LIMIT=50
DEPTHLENS_TRACE_SAMPLE_RATE=1.0
```

### Server

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` locally, `0.0.0.0` in Docker | ASGI bind host |
| `PORT` | `8765` | ASGI port |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `DEBUG` | `false` | Enables FastAPI debug mode (detailed error responses) |
| `WEB_CONCURRENCY` | `1` | Uvicorn worker count (Docker only; use 1 for single-GPU inference) |

### Cache

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | unset | Full Redis URL override (takes precedence over individual fields) |
| `REDIS_HOST` | `127.0.0.1` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis logical database |
| `REDIS_PASSWORD` | unset | Optional Redis password |
| `REDIS_SOCKET_TIMEOUT_SECONDS` | `1.5` | Connect/read timeout; keep low to fail fast and fall back to in-memory |
| `REDIS_MAX_CONNECTIONS` | `20` | Connection pool maximum |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `CACHE_MAX_ENTRIES` | `256` | In-memory LRU entry cap |

### Inference

| Variable | Default | Description |
|---|---|---|
| `DEPTHLENS_PRELOAD_MODEL` | `false` | Warm a model in the background after startup |
| `DEPTHLENS_WARMUP_MODEL` | `MiDaS_small` | Model to pre-warm when preload is enabled |
| `DEPTHLENS_WARMUP_DEVICE` | `auto` | Device to pre-warm on |
| `DEPTHLENS_SKIP_WARMUP` | unset | Set to `1` to skip warmup (used in CI/testing) |
| `DEPTHLENS_MAX_DIM` | `1536` | Maximum long image edge before down-scaling |
| `DEPTHLENS_DEFAULT_METRICS` | `fast` | `none`, `fast`, or `full` |
| `DEPTHLENS_DEFAULT_OUTPUTS` | `color` | `color`, `gray`, or `color,gray` |
| `INFERENCE_MAX_CONCURRENCY` | `2` | Max concurrent inference operations (asyncio semaphore) |
| `DEPTHLENS_OBSERVABILITY_ENABLED` | `true` | Enable local observability snapshots and instrumentation |
| `DEPTHLENS_PROMETHEUS_ENABLED` | `true` | Enable `/metrics` Prometheus exposition when `prometheus-client` is available |
| `DEPTHLENS_TELEMETRY_MAX_EVENTS` | `200` | Bounded recent HTTP/inference event history size |
| `DEPTHLENS_TRACE_HISTORY_LIMIT` | `200` | Bounded trace span history size |
| `DEPTHLENS_CRASH_HISTORY_LIMIT` | `100` | Bounded sanitized crash history size |
| `DEPTHLENS_BENCHMARK_HISTORY_LIMIT` | `50` | Bounded benchmark history size |
| `DEPTHLENS_TRACE_SAMPLE_RATE` | `1.0` | Trace sampling ratio from `0.0` to `1.0` |
| `ORT_INTRA_OP_NUM_THREADS` | CPU-dependent / Docker `2` | ONNX Runtime intra-op thread pool |
| `ORT_INTER_OP_NUM_THREADS` | `1` | ONNX Runtime inter-op thread pool |

### Paths

| Variable | Default | Description |
|---|---|---|
| `DEPTHLENS_BACKEND_PORT` | `8765` | Electron backend port hint (read before spawning uvicorn) |
| `DEPTHLENSPRO_MODEL_DIR` | unset | Custom model directory; ONNX files searched in `{dir}/onnx/` |
| `DEPTHLENS_ONNX_DIR` | unset | Direct ONNX directory override |
| `ONNX_WEIGHTS_DIR` | unset | Legacy ONNX directory (lowest priority) |
| `DEPTHLENS_AUTO_EXPORT_ONNX` | `false` | Auto-export ONNX during benchmark when weights are missing |

### Observability Privacy

Telemetry is local-only: DepthLens Pro does not send analytics to cloud services or external telemetry endpoints. Histories are bounded in process memory, Prometheus labels intentionally avoid high-cardinality user data, and telemetry avoids raw images, base64 payloads, uploaded filenames, image hashes, cache keys, local full paths, and private exception details.

### CI / Test Flags

| Variable | Purpose |
|---|---|
| `TESTING=1` | Lightweight test mode; skips warmup and model downloads |
| `CI=1` | CI marker used by test fixtures |
| `CODEX_ENV=1` | Automation/sandboxed environment marker |
| `DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1` | Prevents torch.hub from downloading weights (used in offline CI) |

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---


## Security and timeout settings

| Variable | Default | Description |
| --- | --- | --- |
| `DEPTHLENS_ROUTE_TIMEOUT_SECONDS` | `300` | Generous timeout for estimate, compare, detect, and reconstruct route work. Timeouts return `REQUEST_TIMEOUT` except `/reconstruct`, which returns `RECONSTRUCTION_TIMEOUT`. |
| `DEPTHLENS_BATCH_ITEM_TIMEOUT_SECONDS` | `300` | Per-item timeout for batch processing so one slow image produces a structured per-item `REQUEST_TIMEOUT` without altering the batch response contract. |
| `DEPTHLENS_CORS_ALLOWED_ORIGINS` | empty | Comma-separated additional browser origins allowed by CORS. Localhost/127.0.0.1/file-null flows are allowed by default. |
| `DEPTHLENS_CORS_ALLOW_ALL` | `false` | Development-only wildcard CORS escape hatch; credentials remain disabled. |

Runtime installs use `backend/requirements.txt`. Development and CI quality checks use `backend/requirements-dev.txt`, which includes runtime dependencies plus Black, Ruff, mypy, and pytest.

See [resource path contract](resource-path-contract.md) for packaged/development resource names shared by backend, Electron, and verification scripts.
