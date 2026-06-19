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
  run python - <<'PY'
from pathlib import Path
readme = Path('README.md')
contrib = Path('CONTRIBUTING.md')
errors = []
if not readme.is_file(): errors.append('README.md is missing')
if not contrib.is_file(): errors.append('CONTRIBUTING.md is missing')
text = '\n'.join(p.read_text(encoding='utf-8') for p in (readme, contrib) if p.is_file()).lower()
for command in ['scripts/ci.sh workflow-policy','scripts/ci.sh docs-contract','scripts/ci.sh backend-quality','scripts/ci.sh electron-contract']:
    if command not in text:
        errors.append(f'document local CI command: {command}')
if 'scripts/ci.sh all' not in text:
    errors.append('document local CI command: scripts/ci.sh all')
if not ('docker' in text and ('optional/manual' in text or ('optional' in text and 'manual' in text)) and 'required ci' in text):
    errors.append('document that Docker builds are optional/manual and not part of required CI')
if errors:
    for error in errors:
        print(f'::error::{error}')
    raise SystemExit(1)
print('Documentation CI contract passed.')
PY
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
  mkdir -p "$tmp_root/backend" "$tmp_root/frontend" "$tmp_root/venv/bin"
  touch "$tmp_root/backend/app.py" "$tmp_root/frontend/index.html" "$tmp_root/venv/bin/python"
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
