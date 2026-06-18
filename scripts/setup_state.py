#!/usr/bin/env python3
"""Setup report helpers for DepthLens Pro."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / ".depthlens"
REPORT_PATH = REPORT_DIR / "setup-report.json"


def write_setup_report(report: dict[str, Any], path: Path = REPORT_PATH) -> Path:
    """Write the machine-readable setup report and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
