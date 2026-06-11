#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/setup-macos.sh
(cd electron-app && npm run verify:resources -- --mode native --onnx optional && npm run build:mac:arm64)
