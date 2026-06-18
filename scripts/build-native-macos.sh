#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=arm64; ONNX_VERIFY_MODE="optional"; ONNX_MODELS="midas_small"; AUTO_SETUP=0
while (($#)); do case "$1" in --arch) shift; ARCH="${1:-arm64}";; --with-onnx) ONNX_VERIFY_MODE="require-all"; ONNX_MODELS="all";; --without-onnx) ONNX_VERIFY_MODE="off";; --onnx-models) shift; ONNX_MODELS="${1:-midas_small}"; [[ "$ONNX_MODELS" == all ]] && ONNX_VERIFY_MODE="require-all" || ONNX_VERIFY_MODE="required";; --auto-setup) AUTO_SETUP=1;; *) echo "Unknown option: $1" >&2; exit 2;; esac; shift || true; done
[[ "$ARCH" == "arm64" ]] || { echo "Unsupported macOS architecture: $ARCH. DepthLens Pro supports macOS Apple Silicon arm64 only." >&2; exit 1; }
echo "[DepthLens] Step 3 Build: platform=macOS arch=arm64 onnx=$ONNX_VERIFY_MODE models=$ONNX_MODELS resource_root=$(pwd)"
if [[ "$AUTO_SETUP" == 1 ]]; then echo "[DepthLens] --auto-setup requested; running setup explicitly."; scripts/setup-macos.sh $([[ "$ONNX_VERIFY_MODE" == require-all ]] && echo --with-onnx --onnx-models all || echo --without-onnx); fi
echo "[DepthLens] Verifying repo resources before packaging (no downloads)."
(cd electron-app && node scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS" ..) || { echo "Run npm run setup:mac (standard) or npm run setup:mac:onnx (ONNX) first." >&2; exit 1; }
(cd electron-app && npm run clean:dist)
echo "[DepthLens] Packaging macOS arm64..."; (cd electron-app && npm run build:mac:arm64:raw)
echo "[DepthLens] Verifying packaged macOS arm64 resources..."; (cd electron-app && node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --torch-cache required --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS")
echo "[DepthLens] SUCCESS macOS arm64 build. Outputs under electron-app/dist"
