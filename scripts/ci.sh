#!/usr/bin/env bash
set -Eeuo pipefail

export CI=1
export TESTING=1
export DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1
export DEPTHLENS_SKIP_WARMUP=1
export DEPTHLENS_CACHE_BACKEND=memory
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

section() { printf '\n==> %s\n' "$*"; }
run() { echo "+ $*"; "$@"; }

docs_contract() {
  section "docs-contract"
  run python scripts/validate_docs.py
}


workflow_policy() { section "workflow-policy"; run python scripts/validate_workflows.py; }
backend_quality() {
  section "backend-quality"
  run python -m black --check .
  run python -m ruff check .
  run python -m mypy backend/
  run python -m pytest -q
}
electron_contract() {
  section "electron-contract"
  run npm --prefix electron-app test
  local tmp_root
  tmp_root="$(mktemp -d)"
  mkdir -p "$tmp_root/backend" "$tmp_root/frontend/js" "$tmp_root/venv/bin"
  touch "$tmp_root/backend/app.py" "$tmp_root/frontend/js/charts.js" "$tmp_root/venv/bin/python"
  cat > "$tmp_root/frontend/index.html" <<'HTML'
<canvas id="latencyChart"></canvas><canvas id="benchmarkChart"></canvas><canvas id="observabilityChart"></canvas><canvas id="compareChart"></canvas>
HTML
  run node electron-app/scripts/verify-resources.js --root-kind repo --mode basic --torch-cache off --detector-cache off --onnx off --json "$tmp_root"
}

case "${1:-}" in
  workflow-policy) workflow_policy ;;
  docs-contract) docs_contract ;;
  backend-quality) backend_quality ;;
  electron-contract) electron_contract ;;
  all) workflow_policy; docs_contract; backend_quality; electron_contract ;;
  *) echo "Usage: $0 {workflow-policy|docs-contract|backend-quality|electron-contract|all}" >&2; exit 2 ;;
esac
