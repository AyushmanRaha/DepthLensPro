# System Design Decisions

[← Back to README](../README.md)

The system design favors transparent local execution over cloud dependency. These decisions keep the desktop app useful on typical developer and reviewer machines while still supporting optional acceleration and reproducible backend workflows.

### Caching

Cache entries are keyed by the model, colormap, device, metrics mode, requested outputs, maximum image dimension, and an image-content hash. Redis is used when configured because it provides TTL-based shared cache behavior across backend workers or repeated sessions. If Redis is not available, the in-memory LRU fallback keeps the app usable without any external service. Ground truth uploads bypass cache so benchmark and evaluation results are recomputed from the current GT input instead of reusing stale image-only outputs.

The tradeoff is that caching improves repeated runs and interactive iteration, but image-specific outputs and benchmark correctness require careful invalidation and bypass rules.

### Concurrency

FastAPI routes can receive multiple requests, but `INFERENCE_MAX_CONCURRENCY` limits how many inference jobs are dispatched at once. Model/device forward locks protect PyTorch and ONNX Runtime session calls for the same model/device pair, while preprocessing and non-critical work can still overlap where safe. This separates request-level concurrency from runtime safety for model execution.

The tradeoff is that maximum throughput is intentionally balanced against local machine stability, GPU/CPU memory pressure, and backend safety.

### Fallback and resilience

Redis is optional; Redis failures should not make local inference fail. Backend health endpoints separate lightweight liveness from deeper readiness and diagnostic checks, so the desktop shell can distinguish "process is reachable" from "all optional runtimes are ready." The Electron main process owns backend startup, port checks, and process lifecycle. Settings persistence uses safer local persistence behavior with recovery and fallback paths.

The tradeoff is that local-first desktop apps need graceful degradation because users may not have Redis, ONNX files, GPU drivers, or identical platform environments.

### Packaging and deployment

Native Electron builds are optimized for the desktop user experience. Docker Compose supports backend + Redis workflows for reproducible backend use without making Docker required for normal desktop inference. ONNX files are locally generated and validated rather than blindly committed, and platform/resource/ONNX resolvers keep packaging logic centralized.

The tradeoff is that packaged desktop convenience must be balanced with large ML assets, platform-specific native dependencies, and optional acceleration providers.

### Local-first privacy

Images are processed on localhost, and no cloud upload or API key is required for normal inference. Telemetry and observability should avoid raw images, image hashes, local paths, filenames, or high-cardinality user data so diagnostics stay useful without exposing private inputs.

The tradeoff is that privacy is prioritized over cloud scalability.

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---
