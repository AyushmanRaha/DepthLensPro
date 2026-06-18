# Refactor Safety Contract

This contract is the behavior-preservation baseline for internal refactor phases. Refactors may improve organization, naming, typing, and testability, but they must not change the public product contract unless a later task explicitly asks for that change.

## No feature/UI changes

- Do not add, remove, rename, restyle, or reorder user-facing UI elements as part of structural refactors.
- Do not change renderer behavior, persistence semantics, security policy, resource checks, platform targets, or packaged resource layout except to preserve existing behavior with clearer structure.
- Do not change model choices, colormap choices, benchmark behavior, cache behavior, diagnostics, or export options.

## API compatibility rule

- Preserve all public endpoint paths, HTTP methods, status codes, request field names, default values, response field names, and response shapes.
- Preserve compatibility aliases such as `/benchmark` and `/api/benchmark`, `/reconstruct` and `/api/reconstruct`, and `/detect` and `/api/detect`.
- `/live` must remain lightweight and dependency-safe. It must continue to return at least `service`, `status`, `version`, `pid`, `timestamp`, and `uptime_seconds` without loading ML models or performing readiness checks.
- The canonical model IDs remain `midas_small`, `dpt_hybrid`, and `dpt_large`.
- The supported colormaps remain `inferno`, `plasma`, `viridis`, `magma`, `jet`, `hot`, `bone`, and `turbo`.

## Install/build compatibility rule

The public native installation workflow remains exactly four steps for both standard and ONNX builds:

1. Clone the repository.
2. Run the platform setup command.
3. Run the platform build command.
4. Run the platform launch command.

Standard setup/build commands must not require ONNX files. ONNX setup/build commands must continue to request all ONNX models with the existing `--with-onnx`/`--onnx-models all`, `-WithOnnx`/`-OnnxModels all`, or equivalent verification flags. Build scripts must continue to verify repo resources before raw packaging and packaged resources after raw packaging.

## Supported platform matrix

| Platform | Architectures | Standard setup | ONNX setup | Standard build | ONNX build | Launch |
| --- | --- | --- | --- | --- | --- | --- |
| macOS | arm64 | `setup:mac` | `setup:mac:onnx` | `build:mac:arm64` | `build:mac:arm64:onnx` | `launch:mac` |
| Windows | x64, arm64 | `setup:win` | `setup:win:onnx` | `build:win:x64`, `build:win:arm64` | `build:win:x64:onnx`, `build:win:arm64:onnx` | `launch:win` |
| Linux | x64, arm64 | `setup:linux` | `setup:linux:onnx` | `build:linux:x64`, `build:linux:arm64` | `build:linux:x64:onnx`, `build:linux:arm64:onnx` | `launch:linux` |

## Required verification commands after every phase

Run these commands after every refactor phase unless a task explicitly narrows verification:

```bash
black --check .
ruff check .
mypy backend/
pytest
cd electron-app && npm test
```

Additional resource checks are phase gates when packaging or install/build scripts are touched:

```bash
npm run verify:resources
npm run verify:onnx
```

Run `npm run verify:onnx` only when ONNX files exist locally or the phase explicitly concerns ONNX validation.

## Files allowed to change in later phases

Later refactor phases should keep changes small and reviewable. Allowed changes are limited to:

- Documentation: `README.md`, `docs/**`, and comments/docstrings that clarify existing behavior.
- Tests and non-runtime verification helpers: `backend/tests/**`, Electron `test-*.js` files, and script-level verification helpers that are not part of runtime behavior.
- Internal backend organization: `backend/**` may be reorganized only when API routes, request/response contracts, cache behavior, model behavior, and install/build behavior remain unchanged and covered by tests.
- Internal Electron organization: `electron-app/**` may be reorganized only when UI, preload/main process public behavior, security policy, resource checks, persistence schema, platform targets, and build scripts remain unchanged and covered by tests.
- Installer/build scripts: `scripts/**`, root `package.json`, and `electron-app/package.json` may change only to preserve or harden the existing clone → setup → build → launch workflow without renaming public commands.

Runtime behavior changes, endpoint contract changes, command renames, UI changes, dependency-download behavior changes, and packaging layout changes are out of scope unless explicitly requested.
