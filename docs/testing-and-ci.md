# Testing & CI

[← Back to README](../README.md)

### Run Local Checks

Install backend dependencies before full pytest collection. Use the normal four-step workflow (`clone → setup → build
→ launch`) and run the appropriate setup command first, for example
`npm run setup:<platform>` for standard builds or `npm run setup:<platform>:onnx`
when validating the required ONNX files (`midas_small.onnx`, `dpt_hybrid.onnx`,
and `dpt_large.onnx`). Standard setup/build does not require ONNX generation.

```bash
python -m black --check .
python -m ruff check .
python -m mypy backend/
python -m pytest backend/tests/test_install_contract.py
python -m pytest

cd electron-app
npm test
cd ..
```

Or as a single pipeline:

```bash
python -m black --check . && python -m ruff check . && python -m mypy backend/ && python -m pytest backend/tests/test_install_contract.py && python -m pytest && cd electron-app && npm test && cd ..
```

### Useful Test Commands

```bash
# Backend tests only
pytest backend/tests/

# One test file with verbose output
pytest backend/tests/test_routes.py -v

# Electron lightweight security and resource tests
cd electron-app && npm test
```

### CI Policy

`ci-passed` is the only required branch-protection check. It always appears, summarizes the dynamic jobs, and fails when a required job failed, was cancelled, or was skipped unexpectedly. PR CI is intentionally dynamic so ordinary documentation and Codex-created branches do not run unnecessary expensive checks.

- Docs-only PRs run the fast docs/policy contract checks only.
- Backend changes run `backend-quality`.
- Electron or frontend changes run `electron-contract`.
- Workflow/tooling changes run `workflow-policy` and the relevant contract checks.
- Pushes to `main` and manual `workflow_dispatch` runs are conservative/full CI runs.
- Docker support files remain in the repo, but Docker builds are currently optional/manual and intentionally removed from required CI because Docker is not an active deployment path right now. Docker CI can be reintroduced later as a separate optional/manual workflow if Docker becomes active again.

Local CI entry points:

```bash
scripts/ci.sh workflow-policy
scripts/ci.sh docs-contract
scripts/ci.sh backend-quality
scripts/ci.sh electron-contract
scripts/ci.sh all
```

The test suite covers API behaviour, cache serialisation safety (no pickle deserialization), ONNX fallback paths, reconstruction logic, packaging verification, and Electron security policies, frontend/backend request contracts, and Electron settings IPC validation — without requiring a GPU, Redis instance, Docker daemon, or real model weights.

#### What the Tests Stub

- **torch / cv2** — fully stubbed via `conftest.py` and `monkeypatch`; no GPU or system OpenCV library required
- **ONNX Runtime** — stubbed per-test with `sys.modules` injection
- **Redis** — disabled via `monkeypatch.setattr(cache_service, "redis", None)`
- **Model downloads** — prevented via `DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1`
- **Warmup** — skipped via `TESTING=1`

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---


## Dynamic CI gate policy

Changed-file detection treats release-affecting files as first-class CI inputs. Backend requirements, Python tooling, setup/build/launch/prefetch scripts, Docker descriptors, resource path policy docs, Electron package files, frontend assets, CI helper scripts, `.gitignore`, and `SECURITY.md` now trigger backend, Electron, docs, workflow, or full gates as appropriate. `ci-passed` remains the branch-protection aggregator and fails when a required dynamic job fails, is cancelled, or is unexpectedly skipped.

`docs-contract` runs `scripts/validate_docs.py` without network access. It validates local Markdown links/images, referenced Electron npm scripts, documented API route declarations, stale engineering/refactor doc links, and root lockfile hygiene. Docker remains optional/manual; required CI does not run Docker builds, Docker Compose, GPU checks, model downloads, or long ONNX exports.

Install development check dependencies with:

```bash
python -m pip install -r backend/requirements-dev.txt
```

Do not run `npm install` at the repository root. The tracked npm lockfile belongs under `electron-app/package-lock.json`.


Electron frontend contract tests also verify chart canvas presence, script ordering, no runtime chart CDN/placeholder dependency, and first-party chart rendering with lightweight fake Canvas 2D contexts.

- CI covers adaptive engine selector behavior, benchmark warmup/recommendation metadata, and frontend contract coverage for the new runtime controls.


### Runtime regression coverage

Tests cover import-light `/live`, runtime activity counters, Compare partial failures, ONNX prefer/strict fallback copy, DPT Large opt-in contracts, detector loaded/detecting/no-object UI states, and resource verification modes. Keep these tests deterministic with mocked models/sessions and downloads disabled.
