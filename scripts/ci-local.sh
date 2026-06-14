#!/usr/bin/env bash
set -euo pipefail

export CI=1
export TESTING=1
export DEPTHLENS_SKIP_WARMUP=1
export DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1
export DEPTHLENS_CACHE_BACKEND=memory
export PYTHONUNBUFFERED=1

black --check .
ruff check .
mypy backend/
pytest backend/tests -q

(
  cd electron-app
  npm test
  node scripts/verify-resources.js --root-kind repo --mode native --onnx off ..
)

if command -v docker >/dev/null 2>&1; then
  docker compose config
else
  echo "Docker not found; skipping local Docker Compose config check."
fi

echo "Local CI checks passed."
