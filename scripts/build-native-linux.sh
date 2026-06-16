#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="$(uname -m)"; case "$ARCH" in x86_64|amd64) ARCH=x64;; aarch64|arm64) ARCH=arm64;; esac
ONNX_VERIFY_MODE="optional"; ONNX_MODELS="midas_small"; SETUP_ARGS=()
while (($#)); do
  case "$1" in
    --arch) shift; ARCH="${1:-$ARCH}";;
    --with-onnx) ONNX_VERIFY_MODE="require-all"; SETUP_ARGS+=("--with-onnx" "--onnx-models" "all") ;;
    --without-onnx) ONNX_VERIFY_MODE="off"; SETUP_ARGS+=("--without-onnx") ;;
    --onnx-models) shift; ONNX_MODELS="${1:-midas_small}"; SETUP_ARGS+=("--onnx-models" "$ONNX_MODELS"); [[ "$ONNX_MODELS" == all ]] && ONNX_VERIFY_MODE="require-all" || ONNX_VERIFY_MODE="required" ;;
    *) SETUP_ARGS+=("$1") ;;
  esac; shift || true
done
if [[ "$ARCH" != "arm64" && "$ARCH" != "x64" ]]; then echo "Unsupported Linux architecture: $ARCH. Supported: arm64, x64." >&2; exit 1; fi
echo "[DepthLens] Linux native build starting for arch=$ARCH onnx=$ONNX_VERIFY_MODE models=$ONNX_MODELS"
scripts/setup-linux.sh "${SETUP_ARGS[@]}"
(cd electron-app && npm run clean:dist)
echo "[DepthLens] Verifying repo resources before packaging..."
(cd electron-app && node scripts/verify-resources.js --root-kind repo --mode native --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS" ..)
echo "[DepthLens] Packaging Linux $ARCH..."
(cd electron-app && npm run "build:linux:${ARCH}:raw")
echo "[DepthLens] Verifying packaged Linux $ARCH resources..."
(cd electron-app && node scripts/verify-packaged-resources.js --platform linux --arch "$ARCH" --mode native --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS")
echo "[DepthLens] Linux $ARCH native build complete."
