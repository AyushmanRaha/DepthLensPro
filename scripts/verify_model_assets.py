#!/usr/bin/env python3
"""Verify DepthLens Pro model assets without downloading them."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.model_assets import ACTION, model_assets_status  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--json', action='store_true')
args = p.parse_args()
status = model_assets_status()
if args.json:
    print(json.dumps(status, indent=2))
else:
    print(f"Model assets status: {status['status']}")
    print(f"ONNX ready: {status['onnx']['any_ready']} (optional)")
    print(f"PyTorch MiDaS cache ready: {status['pytorch_hub']['midas_repo_cached']} at {status['pytorch_hub']['torch_home']}")
    if not status['model_assets_ready']:
        print(f"FIX: {ACTION}")
raise SystemExit(0 if status['model_assets_ready'] else 1)
