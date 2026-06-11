# DepthLens Pro — Codex Instructions

## Primary rule

Work in small, reviewable phases. Do not make a giant PR unless the user explicitly asks for it.

## Cost and environment constraints

This repository is usually edited through Codex Web on ChatGPT Plus.

Do not intentionally download large ML model weights during normal tests.
Do not run long GPU benchmarks.
Do not require Redis, Docker daemon, CUDA, MPS, XPU, Playwright browsers, or external services unless the user explicitly asks.
Prefer lightweight unit tests with mocks/stubs for torch, ONNX Runtime, Redis, and model downloads.

Assume these environment variables may be present:

```bash
CODEX_ENV=1
CI=1
TESTING=1
DEPTHLENS_SKIP_WARMUP=1
DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1
DEPTHLENS_CACHE_BACKEND=memory
ONNX_WEIGHTS_DIR=/workspace/DepthLensPro/models/onnx

