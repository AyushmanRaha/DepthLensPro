#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/setup-linux.sh
(cd electron-app && npm run clean:dist)
echo "Cleaned previous dist/ output."
(cd electron-app && npm run verify:resources -- --mode native --onnx optional && npm run build:linux:arm64)
