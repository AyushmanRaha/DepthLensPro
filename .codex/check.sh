#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".codex/env.sh" ]; then
  source ".codex/env.sh"
fi

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

echo "== DepthLens Pro Codex check =="
echo "Repository: $ROOT"
echo "Python: $(python --version 2>&1 || true)"
echo "Node: $(node --version 2>&1 || true)"
echo "NPM: $(npm --version 2>&1 || true)"

echo
echo "== Python tests =="
if [ -d "backend/tests" ]; then
  python -m pytest backend/tests
else
  echo "Skipping backend/tests because it does not exist."
fi

if [ -d "tests" ]; then
  python -m pytest tests
else
  echo "Skipping tests because it does not exist."
fi

echo
echo "== Python lint/format/type checks =="
if command -v ruff >/dev/null 2>&1; then
  ruff check .
else
  echo "Skipping ruff because it is not installed."
fi

if command -v black >/dev/null 2>&1; then
  black --check .
else
  echo "Skipping black because it is not installed."
fi

if command -v mypy >/dev/null 2>&1 && [ -d "backend" ]; then
  mypy backend/
else
  echo "Skipping mypy because mypy is unavailable or backend/ does not exist."
fi

echo
echo "== Node checks =="
if [ -f "package.json" ]; then
  if node -e 'const p=require("./package.json"); process.exit(p.scripts && p.scripts.test ? 0 : 1)'; then
    npm test
  else
    echo "Skipping npm test because no test script exists."
  fi

  if node -e 'const p=require("./package.json"); process.exit(p.scripts && p.scripts.lint ? 0 : 1)'; then
    npm run lint
  else
    echo "Skipping npm run lint because no lint script exists."
  fi

  if node -e 'const p=require("./package.json"); process.exit(p.scripts && p.scripts.build ? 0 : 1)'; then
    npm run build
  else
    echo "Skipping npm run build because no build script exists."
  fi
else
  echo "Skipping Node checks because package.json does not exist."
fi

echo
echo "== Docker compose config check =="
if command -v docker >/dev/null 2>&1 && [ -f "docker-compose.yml" ]; then
  docker compose config
else
  echo "Skipping docker compose config because Docker is unavailable or docker-compose.yml does not exist."
fi

echo
echo "== All available checks completed =="
