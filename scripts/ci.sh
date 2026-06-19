#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/ci.sh [backend-quality|electron-contract|docker-build|all]

Runs the same lightweight checks used by GitHub Actions. Install dependencies first:
  python -m pip install -r backend/requirements.txt
  (cd electron-app && npm ci)
USAGE
}

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
run() { log "+ $*"; "$@"; }

export CI="${CI:-1}"
export TESTING="${TESTING:-1}"
export DEPTHLENS_DISABLE_MODEL_DOWNLOADS="${DEPTHLENS_DISABLE_MODEL_DOWNLOADS:-1}"
export DEPTHLENS_SKIP_WARMUP="${DEPTHLENS_SKIP_WARMUP:-1}"
export DEPTHLENS_CACHE_BACKEND="${DEPTHLENS_CACHE_BACKEND:-memory}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

target="${1:-all}"

backend_quality() {
  log "Running backend quality checks"
  run python -m black --check .
  run python -m ruff check .
  run python -m mypy backend/
  run python -m pytest -q
  run python scripts/validate_workflows.py
}

electron_contract() {
  log "Running Electron contract checks"
  run npm --prefix electron-app test

  local tmp_root
  tmp_root="$(mktemp -d)"
  trap 'rm -rf "$tmp_root"' RETURN
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
  log "Building backend Docker image"
  run docker build --pull --tag depthlenspro-backend:ci .
}

case "$target" in
  backend-quality) backend_quality ;;
  electron-contract) electron_contract ;;
  docker-build) docker_build ;;
  all)
    backend_quality
    electron_contract
    docker_build
    ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
