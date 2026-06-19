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

## CI policy and validation before opening a pull request

`ci-passed` is the only required branch-protection check. PR CI is dynamic: docs-only PRs run docs/policy checks only, backend changes run `backend-quality`, Electron/frontend changes run `electron-contract`, and workflow/tooling changes run `workflow-policy` plus the relevant checks. Pushes to `main` and manual `workflow_dispatch` runs are conservative/full.

Docker support remains in the repository, but Docker builds are currently optional/manual and not part of required CI because Docker is not an active deployment path right now. Docker CI can be reintroduced later as a separate optional/manual workflow if Docker becomes active again.

Run the same local entry points used by CI:

```bash
scripts/ci.sh workflow-policy
scripts/ci.sh docs-contract
scripts/ci.sh backend-quality
scripts/ci.sh electron-contract
scripts/ci.sh all
```

## Pull request guidelines

- Preserve API response shapes and frontend compatibility unless an intentional breaking change is discussed first.
- Avoid unrelated UI styling changes when working on backend refactors.
- Keep changes focused and include tests for new or refactored behavior.
- Document any setup or operational changes in the README or related docs.
