#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/setup-macos.sh
(cd electron-app && npm run clean:dist)
echo "Cleaned previous dist/ output."
(cd electron-app && npm run verify:resources:native && npm run build:mac:arm64:raw && npm run verify:packaged:mac)
