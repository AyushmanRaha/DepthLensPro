# Contributing to DepthLens Pro

Thank you for improving DepthLens Pro. This project includes an Electron desktop shell, a static frontend, and a FastAPI backend for monocular depth-estimation inference.

## Local setup

1. Clone the repository.
2. Create and activate a Python 3.12 virtual environment. CI runs Python 3.12, so use the same version when possible.
3. Install backend dependencies:

   ```bash
   python -m pip install -r backend/requirements.txt
   ```

4. Install desktop dependencies if you are working on Electron:

   ```bash
   cd electron-app
   npm ci
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

Run the same checks used by CI from the repository root:

```bash
scripts/ci.sh backend-quality
scripts/ci.sh electron-contract
```

If Docker is available and you are touching backend runtime dependencies or packaging, also run:

```bash
scripts/ci.sh docker-build
```

`scripts/ci.sh all` runs every CI gate locally. Keep the `backend-quality`, `electron-contract`, `docker-build`, and `ci-passed` job names stable in `.github/workflows/ci.yml`; branch protection and the aggregate status check rely on those exact names.

## Pull request guidelines

- Preserve API response shapes and frontend compatibility unless an intentional breaking change is discussed first.
- Avoid unrelated UI styling changes when working on backend refactors.
- Keep changes focused and include tests for new or refactored behavior.
- Document any setup or operational changes in the README or related docs.
