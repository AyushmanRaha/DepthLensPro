"""Local system telemetry helpers for API health responses."""

from __future__ import annotations

import os
import shutil
from typing import Any

MEMORY_PRESSURE_LIMIT_PERCENT = 90.0
DISK_USAGE_LIMIT_PERCENT = 90.0
DISK_TELEMETRY_PATH = "/"


def percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def memory_telemetry() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
            for line in meminfo_file:
                key, raw_value = line.split(":", 1)
                meminfo[key] = int(raw_value.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        meminfo = {}

    if meminfo.get("MemTotal") and meminfo.get("MemAvailable") is not None:
        total_bytes = meminfo["MemTotal"]
        available_bytes = meminfo["MemAvailable"]
    elif hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            total_bytes = page_size * physical_pages
            available_bytes = page_size * available_pages
        except (ValueError, OSError, AttributeError):
            return {
                "status": "unknown",
                "pressure_percent": None,
                "limit_percent": MEMORY_PRESSURE_LIMIT_PERCENT,
            }
    else:
        return {
            "status": "unknown",
            "pressure_percent": None,
            "limit_percent": MEMORY_PRESSURE_LIMIT_PERCENT,
        }

    used_bytes = max(total_bytes - available_bytes, 0)
    pressure_percent = percent(used_bytes, total_bytes)
    return {
        "status": "ok" if pressure_percent < MEMORY_PRESSURE_LIMIT_PERCENT else "degraded",
        "pressure_percent": pressure_percent,
        "limit_percent": MEMORY_PRESSURE_LIMIT_PERCENT,
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "used_bytes": used_bytes,
    }


def disk_telemetry(path: str = DISK_TELEMETRY_PATH) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {
            "status": "unknown",
            "path": path,
            "usage_percent": None,
            "limit_percent": DISK_USAGE_LIMIT_PERCENT,
            "error": str(exc),
        }

    used_bytes = usage.total - usage.free
    usage_percent = percent(used_bytes, usage.total)
    return {
        "status": "ok" if usage_percent < DISK_USAGE_LIMIT_PERCENT else "degraded",
        "path": path,
        "usage_percent": usage_percent,
        "limit_percent": DISK_USAGE_LIMIT_PERCENT,
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "used_bytes": used_bytes,
    }


def telemetry_status(*checks: dict[str, Any]) -> str:
    return "degraded" if any(check.get("status") == "degraded" for check in checks) else "ok"
