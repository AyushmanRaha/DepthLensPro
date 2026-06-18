#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v python3 >/dev/null 2>&1 || { echo "DepthLens setup requires Python 3.10-3.12 on PATH as python3. Install Python/venv, then re-run: npm run setup:linux" >&2; exit 127; }
command -v node >/dev/null 2>&1 || { echo "DepthLens setup requires Node.js on PATH as node. Install Node.js, then re-run: npm run setup:linux" >&2; exit 127; }
command -v npm >/dev/null 2>&1 || { echo "DepthLens setup requires npm on PATH. Install Node.js/npm, then re-run: npm run setup:linux" >&2; exit 127; }
python3 scripts/doctor.py --enforce-arch "$@"
