"""Supported native platform matrix for DepthLens Pro."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlatformTarget:
    os_name: str
    platform_key: str
    arch: str
    supported: bool
    label: str
    reason: str | None = None


def normalize_arch(value: str | None) -> str:
    raw = (value or platform.machine() or "").lower()
    aliases = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x64",
        "x86_64": "x64",
        "x64": "x64",
    }
    return aliases.get(raw, raw)


def normalize_os(value: str | None) -> str:
    raw = (value or platform.system() or "").lower()
    aliases = {
        "darwin": "darwin",
        "macos": "darwin",
        "windows": "win32",
        "win32": "win32",
        "linux": "linux",
    }
    return aliases.get(raw, raw)


SUPPORTED_TARGETS = {
    ("darwin", "arm64"),
    ("win32", "arm64"),
    ("win32", "x64"),
    ("linux", "arm64"),
    ("linux", "x64"),
}


def platform_label(os_name: str, arch: str) -> str:
    names = {"darwin": "macOS", "win32": "Windows", "linux": "Linux"}
    return f"{names.get(os_name, os_name)} {arch}"


def evaluate_target(os_name: str | None = None, arch: str | None = None) -> PlatformTarget:
    os_key = normalize_os(os_name)
    arch_key = normalize_arch(arch)
    label = platform_label(os_key, arch_key)
    if (os_key, arch_key) in SUPPORTED_TARGETS:
        return PlatformTarget(
            os_name=os_key, platform_key=os_key, arch=arch_key, supported=True, label=label
        )
    if os_key == "darwin" and arch_key in {"x64", "universal", "x86_64", "amd64"}:
        reason = "macOS x64/universal native builds are intentionally unsupported. Use Apple Silicon macOS arm64."
    elif os_key not in {"darwin", "win32", "linux"}:
        reason = f"Unsupported operating system: {os_key}. Supported: macOS arm64, Windows x64/arm64, Linux x64/arm64."
    else:
        reason = f"Unsupported native target {label}. Supported: macOS arm64, Windows x64/arm64, Linux x64/arm64."
    return PlatformTarget(
        os_name=os_key,
        platform_key=os_key,
        arch=arch_key,
        supported=False,
        label=label,
        reason=reason,
    )


def current_target() -> PlatformTarget:
    return evaluate_target(platform.system(), platform.machine())


def require_supported(os_name: str | None = None, arch: str | None = None) -> PlatformTarget:
    target = evaluate_target(os_name, arch)
    if not target.supported:
        raise SystemExit(target.reason or f"Unsupported native target {target.label}")
    return target


def supported_matrix() -> list[dict[str, Any]]:
    rows = []
    for os_key, arch in [
        ("darwin", "arm64"),
        ("darwin", "x64"),
        ("win32", "x64"),
        ("win32", "arm64"),
        ("linux", "x64"),
        ("linux", "arm64"),
    ]:
        target = evaluate_target(os_key, arch)
        rows.append(
            {
                "platform": os_key,
                "arch": arch,
                "label": target.label,
                "supported": target.supported,
                "reason": target.reason,
            }
        )
    return rows
