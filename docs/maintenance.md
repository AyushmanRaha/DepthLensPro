# DepthLens Pro Maintenance Guide

This guide describes the safe path for future maintainers to extend the refactored codebase without changing public behavior accidentally. Keep changes small and verify each layer before combining them.

## Public behavior compatibility

- Do not rename or break public API endpoints, request fields, status-code expectations, or response shapes without an intentional compatibility phase. Additive response fields should be documented.
- Keep `/live` lightweight and dependency-safe; it must not load ML models or perform deep readiness checks.
- Keep canonical model IDs stable: `midas_small`, `dpt_hybrid`, and `dpt_large`.
- Keep supported colormaps stable unless an intentional compatibility change is planned: `inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, and `turbo`.
- Preserve the public native workflow: clone the repository, run setup, build the app, and launch it.
- Standard setup/build must not require ONNX files. ONNX setup/build must require all supported ONNX files: `midas_small.onnx`, `dpt_hybrid.onnx`, and `dpt_large.onnx`.
- Keep current native platform support explicit: macOS arm64 only; Windows x64/arm64; Linux x64/arm64. macOS x64 and universal builds remain unsupported.
- Keep behavior changes small and covered by lightweight tests that use mocks/stubs instead of real model downloads, Redis, Docker, or GPU-only dependencies.

## Add a model

1. Add the canonical model metadata in `backend/model_registry.py`.
   - Include the public model id, aliases, Torch Hub name, expected ONNX filename, and any model-specific size or transform metadata already represented by the registry.
   - Keep existing ids (`midas_small`, `dpt_hybrid`, `dpt_large`) stable.
2. Update shared constants only when a new public id is intentionally supported.
   - Lightweight cross-layer literals belong in `backend/constants.py`.
   - Path policy and environment override behavior belongs in `backend/core/paths.py`.
3. Teach runtime services about the model through registry data instead of duplicating strings.
   - PyTorch loading lives in `backend/services/model_runtime.py`.
   - ONNX loading and fallback metadata live in `backend/services/onnx_inference.py` and `backend/services/onnx_diagnostics.py`.
   - Export/validation behavior lives in `backend/scripts/export_onnx.py`.
4. Update resource verification only if the set of required native-build assets changes.
   - Electron resource checks live in `electron-app/scripts/verify-resources.js`.
   - Native build wrappers must continue to fail early instead of downloading assets during build.
5. Add lightweight tests with mocks or tiny generated assets.
   - Do not download real model weights in routine tests.
   - Extend `backend/tests/test_model_registry.py`, ONNX guard tests, and install-contract tests as appropriate.

## Add a route

1. Define the route in `backend/api/routes.py` unless it is deliberately a lightweight startup route.
   - `/` and `/live` stay in `backend/api/live.py` to avoid heavy imports.
2. Keep route handlers thin.
   - Request validation belongs in `backend/api/validation.py`.
   - Error-to-payload mapping belongs in `backend/api/errors.py`.
   - Device/readiness state belongs in `backend/api/device_state.py`.
   - Health telemetry belongs in `backend/api/system_telemetry.py`.
   - Business logic belongs in `backend/services/`.
3. Preserve public response shapes for existing routes.
   - If a new field is added, make it additive and document it.
   - Do not rename existing keys or change status codes without an explicit compatibility phase.
4. Add or update route contract tests.
   - Existing public contract coverage lives in `backend/tests/test_routes.py`, `backend/tests/test_refactor_contract.py`, and focused service tests.

## Update installer or build behavior safely

1. Keep the public four-step native flow: clone, setup, build, launch.
2. Keep public root npm script names stable in `package.json`.
3. Maintain standard versus ONNX behavior.
   - Standard setup/build must not require ONNX files.
   - ONNX setup/build must require all three ONNX files: `midas_small.onnx`, `dpt_hybrid.onnx`, and `dpt_large.onnx`.
4. Keep platform support explicit.
   - macOS arm64 is supported.
   - Intel Mac/macOS x64 and macOS universal are unsupported.
   - Windows arm64/x64 and Linux arm64/x64 are supported.
5. Update all related layers together.
   - Setup: `scripts/setup-macos.sh`, `scripts/setup-linux.sh`, `scripts/setup-windows.ps1`, and `scripts/doctor.py`.
   - Build: `scripts/build-native-*.sh`, `scripts/build-native-windows.ps1`, and Electron package scripts.
   - Verification: `electron-app/scripts/verify-resources.js`, packaged resource tests, and install-contract tests.
6. Never hide failures by weakening `pyproject.toml`, `mypy.ini`, CI, or tests.

## Update frontend modules without UI drift

1. Keep `frontend/index.html` script order intentional.
   - Shared state/settings and API helpers load before feature modules.
   - `frontend/js/compat.js` remains the final initialization/compatibility layer.
2. Preserve existing DOM ids, class names, tab order, labels, settings keys, and endpoint calls unless a task explicitly asks for a UI change.
3. Put behavior in the smallest matching module.
   - Uploads: `frontend/js/uploads.js`.
   - Estimate/batch rendering: `frontend/js/inference-ui.js`.
   - Webcam: `frontend/js/webcam.js`.
   - Compare: `frontend/js/compare.js`.
   - Benchmarks: `frontend/js/benchmark.js`.
   - Experiments: `frontend/js/experiments.js`.
   - 3D reconstruction: `frontend/js/reconstruction.js`.
   - Settings/toasts/navigation: `frontend/js/notifications.js` and `frontend/js/settings.js`.
4. Run Electron frontend-contract tests after changes.
   - These tests check script ordering, expected modules, security defaults, platform support, and persistence schema behavior.

## Run the full test matrix

Use these commands before handing off a phase:

```bash
python -m black --check .
python -m ruff check .
python -m mypy backend/
python -m pytest backend/tests/test_install_contract.py
python -m pytest
cd electron-app && npm test
cd .. && npm run verify:resources
```

Run ONNX validation only when all three local ONNX files exist or the phase explicitly changes ONNX behavior:

```bash
npm run verify:onnx
```


## Resource and dependency hygiene

Keep backend, Electron, setup scripts, and packaged-resource verifiers aligned with [resource path contract](resource-path-contract.md). Do not add a tracked root `package-lock.json`; use `electron-app/package-lock.json` and `npm --prefix electron-app` commands.
