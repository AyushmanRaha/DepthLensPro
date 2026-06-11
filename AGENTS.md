# DepthLens Pro — Codex Instructions

## Primary rule

Work in small, reviewable phases. Do not make a giant PR unless the user explicitly asks for it.

## Cost and environment constraints

This repository is usually edited through Codex Web on ChatGPT Plus.

Do not intentionally download large ML model weights during normal tests.
Do not run long GPU benchmarks.
Do not require Redis, Docker daemon, CUDA, MPS, XPU, Playwright browsers, or external services unless the user explicitly asks.
Prefer lightweight unit tests with mocks/stubs for torch, ONNX Runtime, Redis, and model downloads.

# Codex Agent Instructions

## Project Scope

This repository contains the DepthLens Pro application.

When working on frontend overhaul tasks, Codex must read and follow:

`CODEX_FRONTEND_OVERHAUL.md`

That file is the source of truth for the complete visual and frontend redesign.

## Frontend Overhaul Rules

For the frontend overhaul, Codex may modify only these files:

* `frontend/style.css`
* `frontend/index.html`
* `frontend/script.js`
* `frontend/welcome-anim.js`
* `FRONTEND_OVERHAUL_NOTES.md`

Codex must not modify any other files unless explicitly instructed by the user.

## Strictly Prohibited Files and Directories

Do not modify:

* `backend/`
* `electron-app/`
* Python files
* Electron main/preload/security files
* test files
* `README.md`
* `CONTRIBUTING.md`
* `AGENTS.md`
* `Dockerfile`
* `docker-compose.yml`
* `pyproject.toml`
* `mypy.ini`
* `.github/`
* `.codex/`

If a change outside the allowed frontend files seems necessary, do not make that change. Instead, create or update `FRONTEND_OVERHAUL_NOTES.md` and explain the issue.

## Implementation Requirement

When asked to implement the frontend overhaul:

1. Read `CODEX_FRONTEND_OVERHAUL.md` completely.
2. Implement every requirement exactly as written.
3. Preserve all existing application features and behavior.
4. Do not change API contracts, backend behavior, inference logic, benchmark logic, experiment logic, security policy logic, or tests.
5. Keep all existing IDs, classes, ARIA attributes, data attributes, and structural HTML intact unless the markdown file explicitly allows a specific removal.
6. Remove only decorative emoji, glyph, and icon text nodes as described in the markdown.
7. Use the provided design tokens, theme variables, typography system, spacing system, component rules, animation rules, and responsive rules.
8. Log any unresolved edge cases in `FRONTEND_OVERHAUL_NOTES.md`.

## Validation

After making changes, run all feasible validation commands:

```bash
python -m pytest backend/
black --check .
ruff check .
mypy backend/
cd electron-app && npm run test
cd electron-app && npm run verify:resources
node electron-app/test-security-policy.js
```

If a command cannot run because of missing dependencies or environment limitations, report that clearly. Do not claim a check passed unless it was actually run successfully.

## Final Response Requirements

At the end of the task, summarize:

* Files changed
* Files intentionally not changed
* Whether `FRONTEND_OVERHAUL_NOTES.md` was created or updated
* Validation commands run and their results
* Confirmation that prohibited files were not modified


Assume these environment variables may be present:

```bash
CODEX_ENV=1
CI=1
TESTING=1
DEPTHLENS_SKIP_WARMUP=1
DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1
DEPTHLENS_CACHE_BACKEND=memory
ONNX_WEIGHTS_DIR=/workspace/DepthLensPro/models/onnx

