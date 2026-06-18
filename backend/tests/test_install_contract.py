"""Install/build command contract tests for refactor safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _scripts(path: str) -> dict[str, str]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    scripts = data["scripts"]
    assert isinstance(scripts, dict)
    return {str(key): str(value) for key, value in scripts.items()}


def test_root_setup_scripts_still_exist() -> None:
    scripts = _scripts("package.json")

    for name in (
        "setup:mac",
        "setup:linux",
        "setup:win",
        "setup:mac:onnx",
        "setup:linux:onnx",
        "setup:win:onnx",
    ):
        assert name in scripts


def test_root_build_scripts_still_exist() -> None:
    scripts = _scripts("package.json")

    for name in (
        "build:mac:arm64",
        "build:win:x64",
        "build:win:arm64",
        "build:linux:x64",
        "build:linux:arm64",
        "build:mac:arm64:onnx",
        "build:win:x64:onnx",
        "build:win:arm64:onnx",
        "build:linux:x64:onnx",
        "build:linux:arm64:onnx",
    ):
        assert name in scripts


def test_root_launch_scripts_still_exist() -> None:
    scripts = _scripts("package.json")

    for name in ("launch:mac", "launch:win", "launch:linux"):
        assert name in scripts


def test_standard_build_scripts_do_not_require_onnx() -> None:
    scripts = _scripts("package.json")

    for name in (
        "build:mac:arm64",
        "build:win:x64",
        "build:win:arm64",
        "build:linux:x64",
        "build:linux:arm64",
    ):
        command = scripts[name].lower()
        assert "onnx" not in command
        assert "require-all" not in command


def test_onnx_build_scripts_require_all_onnx_models() -> None:
    scripts = _scripts("package.json")

    for name in (
        "build:mac:arm64:onnx",
        "build:win:x64:onnx",
        "build:win:arm64:onnx",
        "build:linux:x64:onnx",
        "build:linux:arm64:onnx",
    ):
        command = scripts[name].lower()
        assert "onnx" in command
        assert "all" in command


def test_electron_scripts_keep_resource_wrappers() -> None:
    scripts = _scripts("electron-app/package.json")

    raw_builds = {
        "build:mac:arm64": "build:mac:arm64:raw",
        "build:win:arm64": "build:win:arm64:raw",
        "build:win:x64": "build:win:x64:raw",
        "build:linux:arm64": "build:linux:arm64:raw",
        "build:linux:x64": "build:linux:x64:raw",
    }
    for script_name, raw_script in raw_builds.items():
        command = scripts[script_name]
        assert "verify:resources" in command
        assert raw_script in command
        assert "verify:packaged" in command
        assert command.index("verify:resources") < command.index(raw_script)
        assert command.index(raw_script) < command.index("verify:packaged")
