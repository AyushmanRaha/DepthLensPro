# Project Structure

```text
DepthLensPro/
├── README.md
├── package.json                 # Root setup/build/dev scripts
├── backend/                     # FastAPI app, routes, services, model registry, tests
│   ├── api/                     # live/readiness/inference/cache/observability routes
│   ├── services/                # inference, cache, diagnostics, GT, reconstruction
│   └── requirements.txt         # Locked Python dependencies
├── electron-app/                # Electron desktop app, packaging config, contract tests
│   ├── main.js
│   ├── preload.js
│   ├── src/                     # Main-process modules and security/process policies
│   ├── scripts/                 # Resource verification and packaging helpers
│   └── package.json
├── frontend/                    # Renderer HTML/CSS/JavaScript app
├── scripts/                     # Cross-platform setup/build/launch/doctor helpers
├── docs/                        # Documentation, screenshots, architecture assets
│   ├── architecture/
│   └── screenshots/
├── models/                      # Local model caches/artifacts; large assets not committed
├── tests/                       # Backend tests
├── docker-compose.yml
├── Dockerfile
└── .github/workflows/ci.yml
```

Large model assets are intentionally not committed. Standard builds use `models/torch-cache`; ONNX builds additionally validate `models/onnx`.
