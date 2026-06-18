#!/usr/bin/env python3
"""Prefetch TorchVision RGB object detector weights for offline/runtime use."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_CACHE = ROOT / "models" / "torch-cache"


def main() -> int:
    TORCH_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE)

    try:
        from torchvision.models.detection import (  # type: ignore[import-not-found]
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - setup needs an actionable top-level failure
        print(
            "ERROR: RGB detector weights could not be cached. Setup needs network "
            "access to cache RGB detector weights for offline / packaged use.\n"
            f"TORCH_HOME was set to: {TORCH_CACHE}\n"
            f"Underlying error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"RGB object detector weights cached successfully under {TORCH_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
