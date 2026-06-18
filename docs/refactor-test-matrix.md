# Refactor Test Matrix

Use this matrix as the phase gate for behavior-preserving refactors. The goal is to prove that formatting, linting, backend typing, backend behavior, Electron policy tests, and resource verification remain stable before moving to the next phase.

| Gate | Command | Purpose |
| --- | --- | --- |
| Python formatting | `black --check .` | Confirms Python files keep the repository formatter contract. |
| Python linting | `ruff check .` | Confirms lint and import-order rules still pass. |
| Backend typing | `mypy backend/` | Confirms backend type contracts remain valid. |
| Backend tests | `pytest` | Confirms API, service, install, and regression tests pass without heavy model downloads. |
| Electron tests | `cd electron-app && npm test` | Confirms security policy, resource verification, platform target, persistence schema, and build-script contract tests pass. |
| Resource verification | `npm run verify:resources` | Confirms native resource verification still runs from the public root command. |
| ONNX verification | `npm run verify:onnx` | Confirms ONNX validation when ONNX files exist locally or the phase touches ONNX behavior. |

## Standard phase gate

```bash
black --check .
ruff check .
mypy backend/
pytest
cd electron-app && npm test
```

## Packaging or install/build phase gate

```bash
npm run verify:resources
```

## ONNX phase gate

```bash
npm run verify:onnx
```

Run the ONNX gate only when the local ONNX files exist or the phase explicitly changes ONNX verification/setup/build behavior. Do not download large model weights solely to satisfy routine refactor checks.
