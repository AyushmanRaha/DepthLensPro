#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="$(uname -m)"; case "$ARCH" in x86_64|amd64) ARCH=x64;; aarch64|arm64) ARCH=arm64;; esac
ONNX_VERIFY_MODE="optional"; ONNX_MODELS="midas_small"; AUTO_SETUP=0
while (($#)); do case "$1" in --arch) shift; ARCH="${1:-$ARCH}";; --with-onnx) ONNX_VERIFY_MODE="require-all"; ONNX_MODELS="all";; --without-onnx) ONNX_VERIFY_MODE="off";; --onnx-models) shift; ONNX_MODELS="${1:-midas_small}"; [[ "$ONNX_MODELS" == all ]] && ONNX_VERIFY_MODE="require-all" || ONNX_VERIFY_MODE="required";; --auto-setup) AUTO_SETUP=1;; *) echo "Unknown option: $1" >&2; exit 2;; esac; shift || true; done
[[ "$ARCH" == "arm64" || "$ARCH" == "x64" ]] || { echo "Unsupported Linux architecture: $ARCH. Supported: arm64, x64." >&2; exit 1; }
echo "[DepthLens] Step 3 Build: platform=Linux arch=$ARCH onnx=$ONNX_VERIFY_MODE models=$ONNX_MODELS resource_root=$(pwd)"
if [[ "$AUTO_SETUP" == 1 ]]; then echo "[DepthLens] --auto-setup requested; running setup explicitly."; scripts/setup-linux.sh $([[ "$ONNX_VERIFY_MODE" == require-all ]] && echo --with-onnx --onnx-models all || echo --without-onnx); fi
echo "[DepthLens] Verifying repo resources before packaging (no downloads)."
(cd electron-app && node scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS" ..) || { echo "Run npm run setup:linux (standard) or npm run setup:linux:onnx (ONNX) first." >&2; exit 1; }
(cd electron-app && npm run clean:dist)
echo "[DepthLens] Packaging Linux $ARCH..."; (cd electron-app && npm run "build:linux:${ARCH}:raw")
echo "[DepthLens] Verifying packaged Linux $ARCH resources..."; (cd electron-app && node scripts/verify-packaged-resources.js --platform linux --arch "$ARCH" --mode native --torch-cache required --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS")
echo "[DepthLens] SUCCESS Linux $ARCH build. Outputs under electron-app/dist"
