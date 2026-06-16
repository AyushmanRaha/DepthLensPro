#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=arm64; ONNX_VERIFY_MODE="optional"; ONNX_MODELS="midas_small"; SETUP_ARGS=()
while (($#)); do
  case "$1" in
    --arch) shift; ARCH="${1:-arm64}";;
    --with-onnx) ONNX_VERIFY_MODE="require-all"; SETUP_ARGS+=("--with-onnx" "--onnx-models" "all") ;;
    --without-onnx) ONNX_VERIFY_MODE="off"; SETUP_ARGS+=("--without-onnx") ;;
    --onnx-models) shift; ONNX_MODELS="${1:-midas_small}"; SETUP_ARGS+=("--onnx-models" "$ONNX_MODELS"); [[ "$ONNX_MODELS" == all ]] && ONNX_VERIFY_MODE="require-all" || ONNX_VERIFY_MODE="required" ;;
    *) SETUP_ARGS+=("$1") ;;
  esac; shift || true
done
if [[ "$ARCH" != "arm64" ]]; then echo "Unsupported macOS architecture: $ARCH. DepthLens Pro supports macOS Apple Silicon arm64 only; macOS x64/universal are not supported." >&2; exit 1; fi
echo "[DepthLens] macOS native build starting for arch=arm64 onnx=$ONNX_VERIFY_MODE models=$ONNX_MODELS"
scripts/setup-macos.sh "${SETUP_ARGS[@]}"
(cd electron-app && npm run clean:dist)
echo "[DepthLens] Verifying repo resources before packaging..."
(cd electron-app && node scripts/verify-resources.js --root-kind repo --mode native --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS" ..)
echo "[DepthLens] Packaging macOS arm64..."
(cd electron-app && npm run build:mac:arm64:raw)
echo "[DepthLens] Verifying packaged macOS arm64 resources..."
(cd electron-app && node scripts/verify-packaged-resources.js --platform darwin --arch arm64 --mode native --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS")
echo "[DepthLens] macOS arm64 native build complete."
