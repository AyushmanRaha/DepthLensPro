# Refactor Completion Report

This report summarizes the completed behavior-preserving refactor and the contracts that remain intentionally unchanged.

## Changed modules by phase

- Phase 1: Established the behavior-preservation baseline, install contract tests, and phase-gate verification matrix.
- Phase 2: Centralized lightweight constants, model ids, filesystem roots, and ONNX path policy.
- Phase 3: Split backend route helpers for validation, error mapping, readiness state, and telemetry while keeping public routes stable.
- Phase 4: Decomposed inference internals into image I/O, cache keys, runtime execution, ONNX execution, metrics, and typed payload helpers.
- Phase 5: Hardened setup/build/resource verification without changing the four-step native workflow.
- Phase 6: Split Electron main-process lifecycle, path, port, Python resolver, settings, and window helpers while preserving IPC and security defaults.
- Phase 7: Split renderer JavaScript into ordered frontend modules and added local-only observability behavior without UI drift.
- Phase 8: Added reliability and performance hardening for cache-hit metadata, ONNX provider normalization, diagnostics, and bounded startup logs.
- Phase 9: Verified the green baseline before final documentation and CI confidence work.
- Final phase: Updated README structure, added maintenance/debugging docs, documented preserved contracts, and kept CI focused on lightweight checks.

## Preserved public contracts

- Native installation remains clone, setup, build, launch for standard and ONNX builds.
- Public root npm script names remain stable.
- Standard setup/build does not require ONNX files.
- ONNX setup/build requires `midas_small.onnx`, `dpt_hybrid.onnx`, and `dpt_large.onnx`.
- macOS arm64, Windows arm64/x64, and Linux arm64/x64 remain supported.
- Intel Mac/macOS x64 and macOS universal remain unsupported.
- FastAPI public route names, request fields, status-code expectations, and response shapes remain preserved.
- Electron UI behavior, preload boundaries, navigation policy, settings persistence, and security defaults remain preserved.
- Build scripts verify resources and fail early; they do not silently download model assets during build.

## Test commands run for final verification

- `python -m black --check .`
- `python -m ruff check .`
- `python -m mypy backend/`
- `python -m pytest backend/tests/test_install_contract.py`
- `python -m pytest`
- `cd electron-app && npm test`
- `npm run verify:resources`
- `npm run verify:onnx` only when all three ONNX files exist locally.

## Known limitations intentionally unchanged

- Routine tests do not download large ML model weights or require GPU, Redis, Docker, CUDA, MPS, XPU, Playwright browsers, or external services.
- ONNX acceleration remains optional for standard builds and unavailable until valid local ONNX files are generated or supplied.
- macOS Intel/x64 and macOS universal builds remain explicitly unsupported.
- Packaged-app verification depends on artifacts produced by the supported native build scripts.
- Frontend JavaScript keeps the existing browser-script loading model; no bundler migration was introduced.
