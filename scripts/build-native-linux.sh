#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ONNX_VERIFY_MODE="require-all"
ONNX_MODELS="all"
ARCH="${DEPTHLENS_TARGET_ARCH:-$(uname -m)}"
case "$ARCH" in x86_64|amd64|x64) ARCH="x64" ;; aarch64|arm64) ARCH="arm64" ;; esac
SETUP_ARGS=("$@")
while (($#)); do
  case "$1" in
    --with-onnx) ONNX_VERIFY_MODE="required" ;;
    --without-onnx) ONNX_VERIFY_MODE="off" ;;
    --onnx-models)
      shift
      ONNX_MODELS="${1:-midas_small}"
      if [[ "$ONNX_VERIFY_MODE" != "off" ]]; then if [[ "$ONNX_MODELS" == "all" ]]; then ONNX_VERIFY_MODE="require-all"; else ONNX_VERIFY_MODE="required"; fi; fi
      ;;
    --onnx-strict) if [[ "$ONNX_VERIFY_MODE" != "off" ]]; then ONNX_VERIFY_MODE="require-all"; fi ;;
    --arch) shift; ARCH="${1:-$ARCH}" ;;
  esac
  shift || true
done
scripts/setup-linux.sh "${SETUP_ARGS[@]}"
(cd electron-app && npm run clean:dist)
echo "Cleaned previous dist/ output."
(cd electron-app && node scripts/verify-resources.js --root-kind repo --mode native --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS" .. && npm run build:linux:${ARCH}:raw && node scripts/verify-packaged-resources.js --platform linux --arch "$ARCH" --mode native --onnx "$ONNX_VERIFY_MODE" --models "$ONNX_MODELS")
if [[ "$ONNX_VERIFY_MODE" == "off" ]]; then echo "ONNX was intentionally skipped for this Linux ARM package."; fi
