#!/usr/bin/env bash
set -Eeuo pipefail

export CI="${CI:-1}"
export TESTING="${TESTING:-1}"
export DEPTHLENS_DISABLE_MODEL_DOWNLOADS="${DEPTHLENS_DISABLE_MODEL_DOWNLOADS:-1}"
export DEPTHLENS_SKIP_WARMUP="${DEPTHLENS_SKIP_WARMUP:-1}"
export DEPTHLENS_CACHE_BACKEND="${DEPTHLENS_CACHE_BACKEND:-memory}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/ci.sh [backend-quality|electron-contract|docker-build|workflow-policy|all]
EOF
}

section() {
  printf '\n==> %s\n' "$*"
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

backend_quality() {
  section "backend-quality"
  run python -m black --check .
  run python -m ruff check .
  run python -m mypy backend/
  run python -m pytest -q
  run python scripts/validate_workflows.py
}

electron_contract() {
  section "electron-contract"
  run npm --prefix electron-app test

  local tmp_root
  tmp_root="$(mktemp -d)"
  cleanup() {
    rm -rf "$tmp_root"
  }
  trap cleanup RETURN

  mkdir -p "$tmp_root/backend" "$tmp_root/frontend" "$tmp_root/venv/bin"
  touch "$tmp_root/backend/app.py" "$tmp_root/frontend/index.html" "$tmp_root/venv/bin/python"

  run node electron-app/scripts/verify-resources.js \
    --root-kind repo \
    --mode basic \
    --torch-cache off \
    --detector-cache off \
    --onnx off \
    --json \
    "$tmp_root"
}

docker_build() {
  section "docker-build"
  run docker build --pull --tag depthlenspro-backend:ci .
}

workflow_policy() {
  section "workflow-policy"
  run python scripts/validate_workflows.py
}

case "${1:-}" in
  backend-quality) backend_quality ;;
  electron-contract) electron_contract ;;
  docker-build) docker_build ;;
  workflow-policy) workflow_policy ;;
  all)
    backend_quality
    electron_contract
    docker_build
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
