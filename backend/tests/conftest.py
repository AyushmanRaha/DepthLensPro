"""Shared test setup for environments without OpenCV system libraries."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


def _unavailable(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("OpenCV is stubbed for route tests")


cv2_stub = ModuleType("cv2")
cv2_stub.COLORMAP_INFERNO = 0
cv2_stub.COLORMAP_PLASMA = 1
cv2_stub.COLORMAP_VIRIDIS = 2
cv2_stub.COLORMAP_MAGMA = 3
cv2_stub.COLORMAP_JET = 4
cv2_stub.COLORMAP_HOT = 5
cv2_stub.COLORMAP_BONE = 6
cv2_stub.COLORMAP_TURBO = 7
cv2_stub.IMREAD_COLOR = 1
cv2_stub.INTER_AREA = 3
cv2_stub.INTER_NEAREST = 0
cv2_stub.COLOR_BGR2RGB = 4
cv2_stub.COLOR_GRAY2BGR = 8
cv2_stub.COLOR_BGR2GRAY = 6
cv2_stub.CV_64F = 6
cv2_stub.imdecode = _unavailable
cv2_stub.resize = _unavailable
cv2_stub.cvtColor = _unavailable
cv2_stub.applyColorMap = _unavailable
cv2_stub.imencode = _unavailable
cv2_stub.GaussianBlur = _unavailable
cv2_stub.Sobel = _unavailable
sys.modules.setdefault("cv2", cv2_stub)
