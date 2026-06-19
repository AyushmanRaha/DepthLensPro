# Contributing to DepthLens Pro

Thank you for improving DepthLens Pro. This project includes an Electron desktop shell, a static frontend, and a FastAPI backend for monocular depth-estimation inference.

## Local setup

1. Clone the repository.
2. Create and activate a Python 3.10+ virtual environment.
3. Install backend dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

4. Install desktop dependencies if you are working on Electron:

   ```bash
   cd electron-app
   npm install
   ```

## Running the backend

From the repository root:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8765 --reload
```

Existing packaged Electron flows that run from the `backend/` directory can continue to use:

```bash
uvicorn app:app --host 127.0.0.1 --port 8765
```

## Validation before opening a pull request

Run the same shared entrypoints used by GitHub Actions so local failures match CI logs:

```bash
scripts/ci.sh backend-quality
scripts/ci.sh electron-contract
scripts/ci.sh docker-build
scripts/ci.sh all
```

`scripts/ci.sh backend-quality` runs Black, Ruff, mypy, pytest, and the workflow policy validator. `scripts/ci.sh electron-contract` runs the Electron contract tests plus a lightweight resource verification dry run against a temporary fake repo root. Docker is required only for `scripts/ci.sh docker-build` or `scripts/ci.sh all`.

CI and the local entrypoint export safe deterministic defaults (`CI=1`, `TESTING=1`, `DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1`, and `DEPTHLENS_SKIP_WARMUP=1`) so tests do not download ML models, warm model caches, require Redis, or depend on external acceleration services.

## Pull request guidelines

- Preserve API response shapes and frontend compatibility unless an intentional breaking change is discussed first.
- Avoid unrelated UI styling changes when working on backend refactors.
- Keep changes focused and include tests for new or refactored behavior.
- Document any setup or operational changes in the README or related docs.
