# Resource path contract

DepthLens Pro keeps development and packaged resource names aligned across backend, Electron, setup scripts, and package verifiers.

| Resource | Development path | Packaged resource name | Notes |
| --- | --- | --- | --- |
| Backend package | `backend/` | `backend/` | FastAPI app and Python services. |
| Frontend assets | `frontend/` | `frontend/` | Static renderer files, including vendored browser assets. |
| Python venv | `venv/` | `venv/` | Packaged runtime environment uses runtime requirements. |
| Models root | `models/` | `models/` | Holds ONNX files and local model caches. |
| ONNX weights | `models/onnx/` | `models/onnx/` | Optional unless ONNX runtime mode is selected. |
| Torch cache | `models/torch-cache/` | `models/torch-cache/` | Optional local cache; tests must not download weights. |
| Detector cache | `models/detectors/` | `models/detectors/` | Optional local detector assets. |

Packaging checks must verify names and presence without downloading models or requiring Docker/GPU services.
