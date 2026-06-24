# Installation Guide

[← Back to README](../README.md)

| Path | Best for | What runs |
|---|---|---|
| A. Native desktop app | Normal desktop use on supported native desktop systems | Packaged Electron app + embedded backend |
| B. Local development | Editing UI or backend code with hot-reload | Manual uvicorn + Electron dev shell |
| C. Backend only | API testing, scripting, CI, integrations | FastAPI server only (no Electron) |
| D. Docker Compose | Containerised backend with Redis cache | Backend container + Redis container |
| E. ONNX acceleration | Faster inference experiments | Locally generated `.onnx` weight files |

---

### A. Native Desktop App

Platform support is explicit and architecture-specific. The setup entry points are `scripts/setup-macos.sh`, `scripts/setup-linux.sh`, and `scripts/setup-windows.ps1` (also exposed through the Electron npm setup scripts).

| Platform / architecture | Status | Build command |
|---|---|---|
| macOS Apple Silicon arm64 | Supported | `npm run build:mac:arm64` |
| Intel Mac / macOS x64 | Not supported | `npm run build:mac:x64` exits with a clear error |
| macOS universal | Not supported | `npm run build:mac:universal` exits with a clear error |
| Windows arm64 | Supported | `npm run build:win:arm64` |
| Windows x64 | Supported | `npm run build:win:x64` |
| Linux arm64 | Supported | `npm run build:linux:arm64` |
| Linux x64 | Supported | `npm run build:linux:x64` |

Windows arm64 and x64 are supported, Linux arm64 and x64 are supported, and macOS remains Apple Silicon only.

The native workflow is deliberately split into **four repeatable steps per platform**: clone, setup, build, and launch. Setup is the only normal step that performs heavyweight dependency installs or model downloads. Standard setup installs the Python venv, backend dependencies, Electron dependencies, PyTorch MiDaS Torch Hub cache, and detector weights; it does **not** generate ONNX by default. ONNX setup adds export/validation for all three files in `models/onnx`: `midas_small.onnx`, `dpt_hybrid.onnx`, and `dpt_large.onnx`. Standard builds require `models/torch-cache` and treat ONNX as optional; ONNX builds require both the PyTorch cache and all three ONNX files.

#### macOS — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac # Install macOS dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:mac:arm64 # Build the macOS Apple Silicon native app
npm run launch:mac # Launch the packaged macOS app
```

</details>

#### macOS — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:mac:onnx # Install macOS dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:mac:arm64:onnx # Build the macOS Apple Silicon native app with ONNX resources required
npm run launch:mac # Launch the packaged macOS app
```

</details>

#### Windows ARM64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:win:arm64 # Build the Windows ARM64 native app
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Windows ARM64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:win:arm64:onnx # Build the Windows ARM64 native app with ONNX resources required
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Windows x86/x64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win # Install Windows dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:win:x64 # Build the Windows x64/x86_64 native app
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Windows x86/x64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```powershell
cd "$HOME\Downloads" # Go to the Downloads folder
if (Test-Path "DepthLensPro\.git") { # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
} elseif (Test-Path "DepthLensPro") { # Check if a non-git folder with the same name exists
Write-Error "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
} else { # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads\DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
} # Finish project folder setup
npm run setup:win:onnx # Install Windows dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:win:x64:onnx # Build the Windows x64/x86_64 native app with ONNX resources required
npm run launch:win # Launch the packaged Windows app
```

</details>

#### Linux ARM64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:linux:arm64 # Build the Linux ARM64 native app
npm run launch:linux # Launch the packaged Linux app
```

</details>

#### Linux ARM64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:linux:arm64:onnx # Build the Linux ARM64 native app with ONNX resources required
npm run launch:linux # Launch the packaged Linux app
```

</details>

#### Linux x86/x64 — Standard native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux # Install Linux dependencies and standard PyTorch model cache
npm run verify:resources # Verify required standard resources before packaging
npm run build:linux:x64 # Build the Linux x64/x86_64 native app
npm run launch:linux # Launch the packaged Linux app
```

</details>

#### Linux x86/x64 — ONNX native build

<details>
<summary>Show clone → setup → build → launch commands</summary>

```bash
cd "$HOME/Downloads" # Go to the Downloads folder
if [ -d "DepthLensPro/.git" ]; then # Check if DepthLensPro is already cloned
cd "DepthLensPro" # Enter the existing project folder
git checkout main # Switch to the main branch
git pull --ff-only # Update the repo without creating merge commits
elif [ -e "DepthLensPro" ]; then # Check if a non-git folder with the same name exists
echo "DepthLensPro exists but is not a Git repo; rename/delete it first." # Warn without overwriting files
exit 1 # Stop safely
else # Run this if the repo is not downloaded yet
git clone https://github.com/AyushmanRaha/DepthLensPro.git "DepthLensPro" # Clone the project into Downloads/DepthLensPro
cd "DepthLensPro" # Enter the cloned project folder
fi # Finish project folder setup
npm run setup:linux:onnx # Install Linux dependencies and generate/validate all ONNX models
npm run verify:onnx:required # Verify that all required ONNX models exist
npm run build:linux:x64:onnx # Build the Linux x64/x86_64 native app with ONNX resources required
npm run launch:linux # Launch the packaged Linux app
```

</details>


#### Setup report and diagnostics

After a successful setup, `scripts/doctor.py` writes a machine-readable report to `.depthlens/setup-report.json`. The report records the detected platform and CPU architecture, selected Python executable, virtualenv path, Node/npm versions, PyTorch MiDaS cache status, detector cache status, ONNX status, and the exact resource verification command that setup used. This file is diagnostic only: build scripts still verify the actual files in `models/torch-cache` and `models/onnx` instead of trusting the report.

Setup is safe to rerun. If pip, npm, MiDaS prefetch, detector prefetch, ONNX handling, or resource verification fails, rerun the platform setup command printed by the failure message. Use the standard setup command for PyTorch builds (`npm run setup:mac`, `npm run setup:linux`, or `npm run setup:win`) and the ONNX setup command when all three ONNX files are required (`npm run setup:mac:onnx`, `npm run setup:linux:onnx`, or `npm run setup:win:onnx`). Passing `--offline` validates existing caches only and does not download model assets; `--onnx-validate-only` validates existing ONNX files and never exports new ones.

#### Setup progress and verification

Setup output is intentionally verbose and streams in real time. Long-running installs and downloads print section headers and commands before they run, for example:

```text
[1/8] Selecting Python
[2/8] Creating or validating venv
[3/8] Upgrading pip/setuptools/wheel/certifi
[4/8] Installing backend dependencies
[5/8] Installing Electron dependencies
[6/8] Caching PyTorch MiDaS assets
[7/8] Handling optional ONNX assets
[8/8] Verifying resources
```

During MiDaS caching, `scripts/prefetch-midas-assets.py` prints the active `TORCH_HOME`, each selected model, whether offline validation is being used, retry counts, and final cache verification. Use these checks after setup:

```bash
npm run verify:resources
node electron-app/scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx optional .
npm run verify:onnx                 # validates existing ONNX files only
node electron-app/scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx require-all --models all .
```

Resource verification also supports structured JSON diagnostics without changing the default human-readable output. The JSON report includes the root kind, verification mode, ONNX mode, model readiness, detector readiness, Python candidate checks, and remediation text. The optional manifest schema in `models/resource-manifest.schema.json` documents the expected resource groups for diagnostics; the verification scripts still inspect the actual filesystem and do not trust the manifest as the sole source of truth.

```bash
node electron-app/scripts/verify-resources.js --json --root-kind repo --mode native --torch-cache required --onnx optional .
npm run verify:resources
npm run verify:onnx
```

Packaged resource verification is performed by the build scripts. To run it manually after packaging, use:

```bash
cd electron-app
node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --torch-cache required --onnx optional
node scripts/verify-packaged-resources.js --platform linux --arch x64 --mode native --torch-cache required --onnx require-all --models all
```

Model assets are intentionally not committed to git. They live in:

| Asset | Repo path | Packaged path | Required for standard builds |
|---|---|---|---|
| PyTorch MiDaS Torch Hub repo, transforms, checkpoints | `models/torch-cache` | `<Resources>/models/torch-cache` | Yes |
| Detector checkpoints | `models/torch-cache/hub/checkpoints` | `<Resources>/models/torch-cache/hub/checkpoints` | Optional feature cache |
| ONNX exports | `models/onnx` | `<Resources>/models/onnx` | No; required only for ONNX builds/benchmarks |

Platform-specific outputs:

| Platform | Output |
|---|---|
| macOS | `electron-app/dist/mac-arm64/DepthLens Pro.app` and `.dmg` |
| Windows | `electron-app/dist/win-arm64-unpacked/` or `electron-app/dist/win-x64-unpacked/` and NSIS installer |
| Linux | `electron-app/dist/*arm64*.AppImage`, `electron-app/dist/*x64*.AppImage`, and unpacked resources when retained by electron-builder |


#### Packaging size preflight and automatic DMG sizing

Every native packaging command now runs `electron-app/scripts/package-size-preflight.js` before Electron Builder starts. The preflight is deterministic and cross-platform: it walks the Electron app, backend, frontend, Python `venv`, `models/torch-cache`, `models/onnx`, and the local Electron runtime cache while ignoring avoidable junk such as `__pycache__`, `.pyc`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.git`, logs, temporary files, and source maps. It prints a per-directory footprint summary, checks free space for the host temp/build/output locations, and fails before packaging if required resources are missing or there is not enough room for a large offline artifact.

For macOS DMG builds, the same preflight computes `dmg.size` automatically instead of relying on a hand-edited hardcoded value. The planned DMG size is the estimated payload size plus a configurable safety margin and filesystem overhead, rounded up to MiB with a sane minimum. The wrapper passes that value to Electron Builder for `npm run build:mac:arm64` and `npm run build:mac:arm64:onnx`, so large bundles containing Electron, the Python virtualenv, Torch/TorchVision, ONNX Runtime, MiDaS Torch cache, and all ONNX exports get a DMG volume sized for the actual payload.

Useful maintainer commands and knobs:

```bash
cd electron-app
npm run preflight:package -- --platform darwin --arch arm64 --onnx require-all --models all
node scripts/package-size-preflight.js --json --platform linux --arch x64 --onnx optional
```

| Environment variable | Default | Purpose |
|---|---:|---|
| `DEPTHLENS_PACKAGE_SIZE_MARGIN` | `0.35` | Fractional safety margin added to the estimated payload before macOS DMG sizing. |
| `DEPTHLENS_DMG_MIN_BYTES` | `2147483648` | Minimum computed macOS DMG size in bytes. |
| `DEPTHLENS_PACKAGE_MIN_FREE_BYTES` | `2147483648` | Minimum host free-space floor for temp/build/output checks; platform payload requirements can raise this automatically. |

Troubleshooting packaging size failures:

- If macOS DMG creation logs `ditto` errors under `/Volumes/DepthLens Pro ...` with `No space left on device`, that often means the temporary mounted DMG image is too small, not that the Mac startup disk is full. Re-run the preflight, inspect the computed `dmg.size`, and either increase `DEPTHLENS_PACKAGE_SIZE_MARGIN`/`DEPTHLENS_DMG_MIN_BYTES` or reduce bundled resources.
- If the preflight reports insufficient host temp/build/output disk space, free space in the listed directory or point the OS temp directory/build output at a larger volume before rerunning the same build command.
- If the preflight reports missing `models/onnx` files for an ONNX build, run `npm run setup:<platform>:onnx` and `npm run verify:onnx:required`; standard non-ONNX builds still require `models/torch-cache` but do not require ONNX exports.
- Large offline builds are expected to be several GiB because they include Electron, the Python runtime, PyTorch/TorchVision, ONNX Runtime, MiDaS Torch cache, and optional ONNX weights such as `dpt_large.onnx`. Avoid deleting required offline model assets; the package filters only exclude caches, logs, compiled Python bytecode, source maps, and temporary/setup diagnostics.

#### No silent downloads during build

The build scripts verify repo resources before packaging and packaged resources after packaging. They do not silently download model assets. If `models/torch-cache` is missing, standard builds fail early with a “run setup first” remediation. If an ONNX build is requested and any ONNX file is missing or empty, the build fails early with the matching `setup:<platform>:onnx` command. Electron packages `models` as extra resources, so packaged resources contain `Resources/models/torch-cache` and, for ONNX builds, `Resources/models/onnx`.

---

### B. Local Development

Run the backend and desktop shell in two separate terminals:

```bash
# Terminal 1
npm run backend:dev
# Equivalent: venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --reload

# Terminal 2
npm run frontend:dev
# Equivalent: cd electron-app && electron .
```

Useful inspection commands:

```bash
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/health      # full diagnostics including device list, ONNX status, memory/disk telemetry
curl http://127.0.0.1:8765/devices     # available compute targets
curl http://127.0.0.1:8765/onnx/status # ONNX weight and provider diagnostics
```

---

### C. Backend Only

```bash
npm run setup
npm run backend:dev
```

Or invoke Uvicorn directly (useful for custom ports or workers):

```bash
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --workers 1
```

The `backend.app` entry point (`backend/app.py`) is a backward-compatible ASGI shim that ensures the repo root is on `sys.path` before importing `backend.main:app`. Both of these are equivalent and interchangeable:

```bash
uvicorn backend.app:app    # repo-root CWD
uvicorn app:app            # backend/ CWD packaged compatibility flow
```

---

### D. Docker Compose

```bash
docker compose up --build          # foreground
docker compose up --build -d       # background
docker compose down                # stop
docker compose down -v             # stop and remove Redis volume
```

Verify:

```bash
curl http://127.0.0.1:8765/live
```

Docker defaults:

| Setting | Default |
|---|---:|
| Backend port | `8765` |
| Redis port | `6379` (internal to Compose network, not exposed) |
| CPU limit | `4.0` cores |
| Memory limit | `8 GB` |
| Shared memory | `8 GB` (needed for PyTorch multiprocessing) |
| Backend user | Non-root `depthlens` (UID/GID created in image) |

The Dockerfile uses a two-stage build: a `builder` stage installs all Python wheels into a venv at `/opt/venv`, and a `runner` stage copies only the venv and the backend package — keeping the final image free of build tools.

---

### E. Optional ONNX Acceleration

ONNX Runtime is entirely optional for standard builds and separate from the required PyTorch MiDaS cache. The app works without ONNX files by using PyTorch MiDaS assets cached under `models/torch-cache`; ONNX builds add `.onnx` files under `models/onnx`.

Generate ONNX files locally:

```bash
# Export the default MiDaS Small graph
venv/bin/python backend/scripts/export_onnx.py --model midas_small --force

# Export all supported models
venv/bin/python backend/scripts/export_onnx.py --all --force

# Validate existing ONNX files without re-exporting
npm run verify:onnx
```

The export script tries two strategies in order:

1. **Legacy `torch.onnx.export`** — standard ONNX opset-17 export with constant-folding.
2. **Dynamo export** — uses `torch.onnx.export(..., dynamo=True)` when the PyTorch version supports it.

If the first strategy produces an invalid graph (checked via `onnx.checker.check_model` and a dummy inference session), the file is quarantined with a `.failed` suffix and the second strategy is tried. This ensures partially-exported files never silently corrupt future benchmark runs.

Expected ONNX location (configurable via `ONNX_WEIGHTS_DIR` or `DEPTHLENSPRO_MODEL_DIR`):

```
models/onnx/
├── midas_small.onnx
├── dpt_hybrid.onnx
└── dpt_large.onnx
```

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---

## Platform-specific v1.0.0 command matrix

Commands below are run from the repository root unless explicitly noted. On Windows, run commands in PowerShell. macOS native builds support Apple Silicon arm64 only; macOS x64 and universal builds intentionally fail with a clear unsupported-architecture message.

### macOS Apple Silicon arm64

Standard native build:

```bash
npm run setup:mac
npm run verify:resources
npm run build:mac:arm64
npm run launch:mac
```

ONNX native build:

```bash
npm run setup:mac:onnx
npm run verify:onnx:required
npm run build:mac:arm64:onnx
npm run launch:mac
```

### Windows ARM64 / x64

Standard native builds (PowerShell):

```powershell
npm run setup:win
npm run verify:resources
npm run build:win:arm64
npm run build:win:x64
npm run launch:win
```

ONNX native builds (PowerShell):

```powershell
npm run setup:win:onnx
npm run verify:onnx:required
npm run build:win:arm64:onnx
npm run build:win:x64:onnx
npm run launch:win
```

### Linux ARM64 / x64

Standard native builds:

```bash
npm run setup:linux
npm run verify:resources
npm run build:linux:arm64
npm run build:linux:x64
npm run launch:linux
```

ONNX native builds:

```bash
npm run setup:linux:onnx
npm run verify:onnx:required
npm run build:linux:arm64:onnx
npm run build:linux:x64:onnx
npm run launch:linux
```

### Terminal-only development

```bash
npm run setup:<platform>
npm run backend:dev
npm run frontend:dev
```

### Backend-only / API usage

```bash
npm run setup
npm run backend:dev
curl http://127.0.0.1:8765/live
curl http://127.0.0.1:8765/ready
```

### Docker

```bash
docker compose up --build
docker compose down
```
