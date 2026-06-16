#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="all"; SKIP_MODELS=0; OFFLINE=0; FORCE=0; TIMEOUT=900; RETRIES=1; DOCTOR_ARGS=()
while (($#)); do
  case "$1" in
    --models) shift; MODELS="${1:-all}" ;;
    --skip-models) SKIP_MODELS=1 ;;
    --offline) OFFLINE=1; DOCTOR_ARGS+=("--offline") ;;
    --force) FORCE=1 ;;
    --timeout-seconds) shift; TIMEOUT="${1:-900}" ;;
    --retries) shift; RETRIES="${1:-1}" ;;
    --without-onnx) DOCTOR_ARGS+=("--without-onnx") ;;
    *) DOCTOR_ARGS+=("$1") ;;
  esac; shift || true
done
echo "[DepthLens] Linux setup: dependencies"
python3 scripts/doctor.py --enforce-arch "${DOCTOR_ARGS[@]}"
if [[ "$SKIP_MODELS" == 1 ]]; then echo "[DepthLens] Skipping model installation; Compare will require MiDaS_small, DPT_Hybrid, DPT_Large before use."; exit 0; fi
PY=".venv/bin/python"; [[ -x "$PY" ]] || PY="venv/bin/python"; [[ -x "$PY" ]] || PY="python3"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torch-cache}"
echo "[DepthLens] Linux setup: installing PyTorch MiDaS models=$MODELS TORCH_HOME=$TORCH_HOME"
ARGS=(scripts/manage_model_assets.py install --models "$MODELS" --timeout-seconds "$TIMEOUT" --retries "$RETRIES")
[[ "$OFFLINE" == 1 ]] && ARGS+=(--offline)
[[ "$FORCE" == 1 ]] && ARGS+=(--force)
"$PY" -u "${ARGS[@]}"
