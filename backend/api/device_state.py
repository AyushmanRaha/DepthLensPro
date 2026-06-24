"""Cached hardware/device state helpers for API routes."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import time
from collections.abc import Callable
from typing import Any

from backend.api.errors import benchmark_timeout, http_error

PROBE_TTL_SECONDS = 10.0
DEVICE_CACHE: dict[str, Any] = {"expires_at": 0.0, "devices": None, "primary": "cpu", "error": None}
ACCEL_CACHE: dict[str, Any] = {"expires_at": 0.0, "checks": None, "error": None, "device_keys": ()}
READINESS_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "device": None,
    "payload": None,
    "error": None,
}


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def fallback_cpu(error: str | None = None) -> dict[str, Any]:
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "System CPU"
    return {
        "cpu": {
            "name": f"CPU · {cpu_name}",
            "hardware_name": cpu_name,
            "type": "cpu",
            "compute_classes": ["cpu"],
            "available": True,
            **({"discovery_error": error} if error else {}),
        }
    }


def cached_devices(
    *,
    available_devices: Callable[[], dict[str, Any]],
    default_device_key: Callable[[], str],
    log: logging.Logger,
    force: bool = False,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    now = time.time()
    if (
        not force
        and DEVICE_CACHE.get("devices") is not None
        and float(DEVICE_CACHE.get("expires_at", 0.0)) > now
    ):
        return (
            DEVICE_CACHE["devices"],
            str(DEVICE_CACHE.get("primary") or "cpu"),
            {"cached": True, "error": DEVICE_CACHE.get("error")},
        )

    started = time.perf_counter()
    error = None
    try:
        devs = available_devices()
        if "cpu" not in devs:
            devs = {**fallback_cpu("device discovery omitted CPU"), **devs}
        primary = default_device_key()
        if primary not in devs:
            primary = "cpu"
    except Exception as exc:
        error = str(exc)
        log.warning("Device discovery degraded: %s", exc)
        devs = fallback_cpu(error)
        primary = "cpu"

    DEVICE_CACHE.update(
        {"expires_at": now + PROBE_TTL_SECONDS, "devices": devs, "primary": primary, "error": error}
    )
    return devs, primary, {"cached": False, "error": error, "duration_ms": elapsed_ms(started)}


def cached_acceleration_checks(
    devs: dict[str, Any],
    *,
    acceleration_checks: Callable[[dict[str, Any]], dict[str, Any]],
    log: logging.Logger,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = time.time()
    device_keys = tuple(sorted(devs))
    if (
        not force
        and ACCEL_CACHE.get("checks") is not None
        and ACCEL_CACHE.get("device_keys") == device_keys
        and float(ACCEL_CACHE.get("expires_at", 0.0)) > now
    ):
        return ACCEL_CACHE["checks"], {"cached": True, "error": ACCEL_CACHE.get("error")}

    started = time.perf_counter()
    error = None
    try:
        checks = acceleration_checks(devs)
    except Exception as exc:
        error = str(exc)
        log.warning("Acceleration probe degraded: %s", exc)
        checks = {
            "cuda": {
                "available": any(k.startswith("cuda:") for k in devs),
                "operational": False,
                "error": error,
            },
            "mps": {"available": "mps" in devs, "operational": False, "error": error},
            "xpu": {
                "available": any(k.startswith("xpu:") for k in devs),
                "operational": False,
                "error": error,
            },
        }
    ACCEL_CACHE.update(
        {
            "expires_at": now + PROBE_TTL_SECONDS,
            "checks": checks,
            "error": error,
            "device_keys": device_keys,
        }
    )
    return checks, {"cached": False, "error": error, "duration_ms": elapsed_ms(started)}


def cached_readiness_payload(
    device: str,
    *,
    readiness_payload: Callable[[str], dict[str, Any]],
    log: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = time.time()
    if (
        READINESS_CACHE.get("payload") is not None
        and READINESS_CACHE.get("device") == device
        and float(READINESS_CACHE.get("expires_at", 0.0)) > now
    ):
        return READINESS_CACHE["payload"], {"cached": True, "error": READINESS_CACHE.get("error")}

    started = time.perf_counter()
    error = None
    try:
        payload = readiness_payload(device)
    except Exception as exc:
        error = str(exc)
        log.warning("Readiness diagnostics degraded: %s", exc)
        payload = {
            "status": "degraded",
            "overall_status": "diagnostics_unavailable",
            "error": error,
        }
    READINESS_CACHE.update(
        {
            "expires_at": now + PROBE_TTL_SECONDS,
            "device": device,
            "payload": payload,
            "error": error,
        }
    )
    return payload, {"cached": False, "error": error, "duration_ms": elapsed_ms(started)}


def validated_device_or_422(
    device: str,
    *,
    cached_devices_func: Callable[..., tuple[dict[str, Any], str, dict[str, Any]]],
    resolve: Callable[[str], Any],
    log: logging.Logger,
) -> str:
    def available_options(force: bool = False) -> list[str]:
        devs, _, _ = cached_devices_func(force=force)
        return [*devs.keys(), "auto"]

    avail = available_options()
    if device not in avail:
        raise http_error(
            422,
            "DEVICE_UNAVAILABLE",
            f"Device '{device}' is unavailable. Options: {avail}",
            field="device",
        )
    try:
        return str(resolve(device))
    except asyncio.TimeoutError as exc:
        log.warning("Benchmark timed out", extra={"device": device})
        raise benchmark_timeout() from exc
    except ValueError as exc:
        refreshed = available_options(force=True)
        if device not in refreshed:
            raise http_error(
                422,
                "DEVICE_UNAVAILABLE",
                f"Device '{device}' is unavailable. Options: {refreshed}",
                field="device",
            ) from exc
        try:
            return str(resolve(device))
        except ValueError as refreshed_exc:
            raise http_error(
                422,
                "DEVICE_VALIDATION_ERROR",
                "Requested device could not be resolved after refreshing available devices.",
                field="device",
            ) from refreshed_exc
