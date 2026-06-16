#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
APP="electron-app/dist/mac-arm64/DepthLens Pro.app"
echo "[DepthLens] Expected macOS app path: $APP"
if [[ ! -d "$APP" ]]; then echo "Packaged app not found. Build first: npm run build:mac:arm64" >&2; exit 1; fi
open "$APP"
