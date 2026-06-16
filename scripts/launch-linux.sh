#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="$(uname -m)"; case "$ARCH" in x86_64|amd64) ARCH=x64;; aarch64|arm64) ARCH=arm64;; esac
APP="electron-app/dist/linux-$ARCH-unpacked/depthlens-pro"
echo "[DepthLens] Expected Linux app path: $APP"
if [[ ! -x "$APP" ]]; then echo "Packaged app not found. Build first: npm run build:linux:$ARCH" >&2; exit 1; fi
exec "$APP"
