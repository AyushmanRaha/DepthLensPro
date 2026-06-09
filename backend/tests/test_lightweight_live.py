"""Liveness endpoint import and latency guarantees."""

from __future__ import annotations

import subprocess
import sys
import time

from fastapi.testclient import TestClient

from backend.main import app


def test_live_response_under_one_second() -> None:
    client = TestClient(app)
    start = time.perf_counter()
    response = client.get("/live")
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 1.0


def test_app_import_does_not_import_heavy_ml_modules() -> None:
    code = """
import sys
import backend.main
for name in ('torch', 'cv2', 'onnxruntime'):
    if name in sys.modules:
        raise SystemExit(f'{name} imported during /live app import')
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr + result.stdout
