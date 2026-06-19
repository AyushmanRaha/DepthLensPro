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

Run the same entrypoints used by CI:

```bash
scripts/ci.sh workflow-policy
scripts/ci.sh docs-contract
scripts/ci.sh backend-quality
scripts/ci.sh electron-contract
scripts/ci.sh docker-build
scripts/ci.sh all
```

PR CI is dynamic. Docs-only changes run docs/workflow checks only, backend changes run backend checks, Electron/frontend changes run Electron checks, and Docker-related changes run the cached Docker build. Pushes to `main` and manual `workflow_dispatch` runs execute full CI. Branch protection should require only the stable `ci-passed` check; the internal jobs are allowed to skip when `detect-changes` marks them irrelevant. Docker may not be available in Codex or local sandboxes, so `scripts/ci.sh docker-build` fails fast locally while GitHub Actions runs Docker when required.

## Pull request guidelines

- Preserve API response shapes and frontend compatibility unless an intentional breaking change is discussed first.
- Avoid unrelated UI styling changes when working on backend refactors.
- Keep changes focused and include tests for new or refactored behavior.
- Document any setup or operational changes in the README or related docs.
