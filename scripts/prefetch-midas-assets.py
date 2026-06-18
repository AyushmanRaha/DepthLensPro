#!/usr/bin/env python3
"""Prefetch/validate PyTorch MiDaS Torch Hub assets with live progress."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.constants import SUPPORTED_ONNX_MODEL_IDS
from backend.core.paths import TORCH_CACHE_ROOT
from backend.model_registry import get_model_spec
from backend.services.model_assets import inspect_model_assets

IDS = SUPPORTED_ONNX_MODEL_IDS


def parse_models(v: str):
    items = [x.strip() for x in v.split(",") if x.strip()]
    if items == ["all"]:
        return list(IDS)
    bad = [x for x in items if x not in IDS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unsupported model(s): {', '.join(bad)}; expected {', '.join(IDS)} or all"
        )
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=parse_models,
        default=list(IDS),
        help="midas_small,dpt_hybrid,dpt_large or all",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--offline", action="store_true", help="validate cache only; do not load/download"
    )
    p.add_argument("--timeout-seconds", type=int, default=900)
    p.add_argument("--retries", type=int, default=2)
    args = p.parse_args()
    models = args.models if isinstance(args.models, list) else parse_models(args.models)
    torch_home = Path(os.environ.get("TORCH_HOME", TORCH_CACHE_ROOT)).resolve()
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[MiDaS assets] TORCH_HOME={torch_home}", flush=True)
    print(
        f"[MiDaS assets] models={','.join(models)} offline={args.offline} "
        f"force={args.force} retries={args.retries} timeout={args.timeout_seconds}s",
        flush=True,
    )
    torch_home.mkdir(parents=True, exist_ok=True)
    if args.offline:
        status = inspect_model_assets(required_models=tuple(models), cache_root=torch_home)
        ready = status["pytorch_hub_cache_ready"]
        print(
            f"[MiDaS assets] offline verification: pytorch_hub_cache_ready={ready}",
            flush=True,
        )
        print(status, flush=True)
        raise SystemExit(0 if ready else 1)
    import torch

    if args.force:
        # Torch Hub will revalidate/load. Do not delete existing cache here.
        print(
            "[MiDaS assets] --force requested; existing cache will be reused where "
            "Torch Hub permits.",
            flush=True,
        )
    start = time.monotonic()
    print("[MiDaS assets] [transforms] loading intel-isl/MiDaS transforms...", flush=True)
    torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
        skip_validation=True,
        force_reload=args.force,
    )
    print("[MiDaS assets] [transforms] OK", flush=True)
    for idx, model_id in enumerate(models, 1):
        spec = get_model_spec(model_id)
        for attempt in range(1, args.retries + 2):
            elapsed = time.monotonic() - start
            if elapsed > args.timeout_seconds:
                raise SystemExit(
                    f"Timed out after {args.timeout_seconds}s while caching {model_id}. "
                    "Re-run setup with --timeout-seconds larger or use --offline to "
                    "validate an existing cache."
                )
            attempt_label = f"{attempt}/{args.retries + 1}"
            print(
                f"[MiDaS assets] [{idx}/{len(models)}] loading/caching {model_id} "
                f"({spec.pytorch_model_name}) attempt {attempt_label}",
                flush=True,
            )
            try:
                model = torch.hub.load(
                    "intel-isl/MiDaS",
                    spec.pytorch_model_name,
                    trust_repo=True,
                    skip_validation=True,
                    force_reload=args.force and attempt == 1,
                )
                model.eval().cpu()
                del model
                print(f"[MiDaS assets] [{model_id}] OK cached", flush=True)
                break
            except Exception as exc:
                print(
                    f"[MiDaS assets] [{model_id}] FAILED: {type(exc).__name__}: {exc}", flush=True
                )
                if attempt > args.retries:
                    raise
                time.sleep(min(10, attempt * 2))
    status = inspect_model_assets(required_models=tuple(models), cache_root=torch_home)
    print(
        f"[MiDaS assets] verification pytorch_hub_cache_ready={status['pytorch_hub_cache_ready']}",
        flush=True,
    )
    if not status["pytorch_hub_cache_ready"]:
        print(status, flush=True)
        raise SystemExit(
            "MiDaS cache incomplete. Re-run setup with network access; do not build "
            "until this is OK."
        )
    print("[MiDaS assets] complete.", flush=True)


if __name__ == "__main__":
    main()
