#!/usr/bin/env bash
set -Eeuo pipefail

export CI=1
export TESTING=1
export DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1
export DEPTHLENS_SKIP_WARMUP=1
export DEPTHLENS_CACHE_BACKEND=memory
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

usage() {
  cat <<'EOF'
Usage: scripts/ci.sh <backend-quality|electron-contract|docker-build|workflow-policy|all>
EOF
}

backend_quality() {
  python -m black --check .
  python -m ruff check .
  python -m mypy backend/
  python -m pytest -q
}

electron_contract() {
  (
    cd electron-app
    npm test
    tmp_root="$(mktemp -d)"
    trap 'rm -rf "$tmp_root"' EXIT
    mkdir -p "$tmp_root/backend" "$tmp_root/frontend" "$tmp_root/venv/bin"
    touch "$tmp_root/backend/app.py" "$tmp_root/frontend/index.html" "$tmp_root/venv/bin/python"
    node scripts/verify-resources.js \
      --root-kind repo \
      --mode basic \
      --torch-cache off \
      --detector-cache off \
      --onnx off \
      --json \
      "$tmp_root"
  )
}

docker_build() {
  docker build --pull --tag depthlenspro-backend:ci .
}

workflow_policy() {
  python scripts/validate_workflows.py
}

case "${1:-}" in
  backend-quality) backend_quality ;;
  electron-contract) electron_contract ;;
  docker-build) docker_build ;;
  workflow-policy) workflow_policy ;;
  all)
    workflow_policy
    backend_quality
    electron_contract
    docker_build
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
