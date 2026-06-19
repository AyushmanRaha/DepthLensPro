#!/usr/bin/env bash
set -Eeuo pipefail

export CI=1 TESTING=1 DEPTHLENS_DISABLE_MODEL_DOWNLOADS=1 DEPTHLENS_SKIP_WARMUP=1 DEPTHLENS_CACHE_BACKEND=memory
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

section() { printf '\n==> %s\n' "$*"; }
run() { printf '+ %q' "$1"; shift; printf ' %q' "$@"; printf '\n'; "$@"; }

backend_quality() {
  section "Backend quality"
  run black python -m black --check .
  run ruff python -m ruff check .
  run mypy python -m mypy backend/
  run pytest python -m pytest -q
}

electron_contract() {
  section "Electron contract"
  if [ ! -d electron-app/node_modules ]; then
    run npm-ci bash -lc 'cd electron-app && npm ci'
  fi
  run npm-test bash -lc 'cd electron-app && npm test'
}

docker_build() {
  section "Docker build"
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed; install/start Docker or let GitHub Actions run docker-build." >&2
    return 127
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is unavailable; docker-build is intentionally not required for non-Docker checks." >&2
    return 127
  fi
  run docker docker buildx build --load --tag depthlenspro-backend:local .
}

workflow_policy() { section "Workflow policy"; run validate python scripts/validate_workflows.py; }

docs_contract() {
  section "Docs contract"
  python - <<'PY'
from pathlib import Path
import sys
readme = Path('README.md').read_text(encoding='utf-8')
contrib = Path('CONTRIBUTING.md').read_text(encoding='utf-8')
errors = []
for anchor in ['## Testing & CI', 'scripts/ci.sh backend-quality', 'ci-passed']:
    if anchor not in readme:
        errors.append(f'README.md must document durable CI contract item: {anchor}')
for command in ['workflow-policy','docs-contract','backend-quality','electron-contract','docker-build','all']:
    text = f'scripts/ci.sh {command}'
    if text not in readme or text not in contrib:
        errors.append(f'README.md and CONTRIBUTING.md must document {text}')
if errors:
    print('\n'.join(f'ERROR: {e}' for e in errors), file=sys.stderr)
    sys.exit(1)
print('Docs contract passed.')
PY
}

case "${1:-}" in
  backend-quality) backend_quality ;;
  electron-contract) electron_contract ;;
  docker-build) docker_build ;;
  workflow-policy) workflow_policy ;;
  docs-contract) docs_contract ;;
  all) workflow_policy; docs_contract; backend_quality; electron_contract; docker_build ;;
  *) echo "Usage: $0 {backend-quality|electron-contract|docker-build|workflow-policy|docs-contract|all}" >&2; exit 2 ;;
esac
