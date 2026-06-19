# API Reference

The FastAPI backend runs locally, normally at `http://127.0.0.1:8765`.

## Endpoint summary

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service identity and API version |
| `GET` | `/live` | Lightweight liveness with `status`, `busy`, `state`, `service`, `version`, `pid`, `timestamp`, and `uptime_seconds` |
| `GET` | `/ready` | Runtime import, model asset, PyTorch cache, ONNX, and inference readiness |
| `GET` | `/health` | Full backend diagnostics |
| `GET` | `/devices` | Available inference devices |
| `GET` | `/models` | Supported canonical model IDs and display metadata |
| `GET` | `/colormaps` | Supported colormap names |
| `GET` | `/onnx/status` | ONNX Runtime provider and model-file status |
| `GET` | `/benchmark`, `/api/benchmark` | Benchmark status and latest benchmark results |
| `GET` | `/cache/metrics` | Cache backend, hit/miss, and fallback metrics |
| `GET` | `/metrics` | Prometheus exposition text |
| `GET` | `/api/observability`, `/observability` | Local observability snapshot |
| `DELETE` | `/cache` | Clear cache |
| `POST` | `/cache/clear` | Clear cache alias for clients that prefer POST |
| `POST` | `/estimate` | Estimate depth for one uploaded image |
| `POST` | `/batch` | Estimate depth for multiple images |
| `POST` | `/api/reconstruct`, `/reconstruct` | Generate approximate point-cloud preview/export data |
| `POST` | `/api/detect`, `/detect` | RGB detection helper route |

## Common curls

```bash
curl http://127.0.0.1:8765/
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/devices
curl http://127.0.0.1:8765/models
curl http://127.0.0.1:8765/colormaps
curl http://127.0.0.1:8765/onnx/status
curl http://127.0.0.1:8765/cache/metrics
curl -X POST http://127.0.0.1:8765/cache/clear
```

## `POST /estimate`

Multipart form endpoint for one image. Common fields include `file`, `model`, `device`, `colormap`, optional output/metric switches, maximum image dimension, and optional ground-truth fields. The response includes encoded depth visualization data, selected model/device metadata, timing, cache metadata, and optional metrics.

```bash
curl -X POST http://127.0.0.1:8765/estimate \
  -F "file=@example.jpg" \
  -F "model=midas_small" \
  -F "device=cpu" \
  -F "colormap=magma"
```

## `POST /batch`

Multipart endpoint for multiple image files with the same inference options. It returns per-image results and errors without changing route behavior.

```bash
curl -X POST http://127.0.0.1:8765/batch \
  -F "files=@one.jpg" \
  -F "files=@two.jpg" \
  -F "model=midas_small"
```

## `POST /reconstruct` and `/api/reconstruct`

Generates approximate 3D point-cloud data from an uploaded image/depth request. Options mirror the app's 3D reconstruction UI and can return preview points plus PLY/OBJ export payloads.

## `POST /detect` and `/api/detect`

Detection helper endpoint used by the local app workflow. It accepts uploaded image data and returns structured local detection results when detector assets are available.

## Cache clearing

Both forms are supported by the backend:

```bash
curl -X DELETE http://127.0.0.1:8765/cache
curl -X POST http://127.0.0.1:8765/cache/clear
```
