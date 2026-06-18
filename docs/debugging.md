# DepthLens Pro Debugging Guide

Use this guide to diagnose common runtime and packaged-app failures without changing public install, API, or UI behavior.

## Backend startup failures

1. Check whether the lightweight endpoint starts.
   ```bash
   cd electron-app && npm run backend:live
   ```
2. Run the backend diagnostic helper for port, process, and endpoint details.
   ```bash
   python scripts/diagnose_backend.py
   ```
3. Inspect Electron backend startup output.
   - Main-process lifecycle code keeps a bounded backend-output tail for remediation messages.
   - Startup orchestration lives in `electron-app/src/main/backend-lifecycle.js`.
4. For import failures, verify dependencies in the active venv.
   ```bash
   python -m pytest backend/tests/test_lightweight_live.py
   ```

## Missing model assets

Standard native builds require the PyTorch MiDaS Torch Hub cache under `models/torch-cache`; ONNX files are optional for standard builds.

1. Verify repo resources.
   ```bash
   npm run verify:resources
   ```
2. If the Torch cache is missing, rerun the platform setup step.
   ```bash
   npm run setup:mac      # or setup:linux / setup:win
   ```
3. Rebuild after setup so packaged resources include the refreshed cache.
   ```bash
   npm run build:mac:arm64
   ```

## ONNX missing or invalid files

ONNX builds and ONNX-only verification require all three files in `models/onnx`:

- `midas_small.onnx`
- `dpt_hybrid.onnx`
- `dpt_large.onnx`

1. For a standard build, missing ONNX files are not fatal.
2. For an ONNX build, rerun ONNX setup for the target platform.
   ```bash
   npm run setup:mac:onnx      # or setup:linux:onnx / setup:win:onnx
   ```
3. Validate existing ONNX files only when they exist locally.
   ```bash
   npm run verify:onnx
   ```
4. Use `npm run verify:onnx:required` to confirm all three ONNX files are present and non-empty without exporting new files.

## Port conflicts

DepthLens Pro probes backend ports before launching and records backend ownership metadata to avoid killing unrelated processes.

1. Run the diagnostic helper.
   ```bash
   python scripts/diagnose_backend.py
   ```
2. Stop a DepthLens-owned backend process.
   ```bash
   cd electron-app && npm run kill:backend
   ```
3. If another process owns the port, stop that process manually or launch DepthLens on a free fallback port.

## Packaged resource failures

Packaged apps must contain the backend, frontend, Python runtime expectations, Torch cache, and optional/required ONNX files according to the build mode.

1. Verify a packaged app after build.
   ```bash
   cd electron-app && npm run verify:packaged
   ```
2. For platform-specific checks, use the matching script.
   ```bash
   cd electron-app && npm run verify:packaged:mac
   cd electron-app && npm run verify:packaged:win
   cd electron-app && npm run verify:packaged:linux
   ```
3. If verification fails, rerun setup and rebuild rather than copying files into an installed app by hand.

## Electron settings corruption

Settings persistence is sanitized and backed up when corruption is detected.

1. Settings schema and backup behavior live in `electron-app/src/main/settings-store.js`.
2. Run the persistence schema test.
   ```bash
   cd electron-app && node test-persistence-schema.js
   ```
3. If a local settings file is corrupt, keep the generated backup for diagnosis and let the app recreate sanitized defaults.
4. Do not relax Electron security defaults to work around settings issues; keep preload/contextBridge boundaries intact.
