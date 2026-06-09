"""Electron packaging and helper script configuration checks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = json.loads((ROOT / "electron-app" / "package.json").read_text())
README = (ROOT / "README.md").read_text()


def test_package_builds_only_arm_desktop_targets() -> None:
    scripts = PACKAGE["scripts"]
    assert "--arm64" in scripts["build:mac:arm64"]
    assert "--arm64" in scripts["build:win:arm64"]
    assert "--arm64" in scripts["build:linux:arm64"]
    for name in ("build:mac:x64", "build:mac:universal", "build:win:x64", "build:linux:x64"):
        assert "unsupported-arch.js" in scripts[name]
    assert PACKAGE["build"]["mac"]["target"][0]["arch"] == ["arm64"]
    assert PACKAGE["build"]["win"]["target"][0]["arch"] == ["arm64"]
    assert PACKAGE["build"]["linux"]["target"][0]["arch"] == ["arm64"]


def test_readme_platform_table_matches_package_scripts() -> None:
    assert "macOS Apple Silicon only" in README
    assert "Windows ARM only" in README
    assert "Linux ARM only" in README
    assert "Intel Mac / macOS x64" in README
    assert "npm run build:mac" in README
    assert "dist/mac-arm64/DepthLens Pro.app" in README


def test_duplicate_app_scan_flags_stale_dist_mac(tmp_path: Path) -> None:
    import os
    import subprocess

    stale = tmp_path / "electron-app" / "dist" / "mac" / "DepthLens Pro.app"
    arm = tmp_path / "electron-app" / "dist" / "mac-arm64" / "DepthLens Pro.app"
    stale.mkdir(parents=True)
    arm.mkdir(parents=True)
    env = {**os.environ, "DEPTHLENS_APP_SCAN_ROOTS": str(tmp_path)}
    result = subprocess.run(
        ["node", str(ROOT / "electron-app" / "scripts" / "scan-apps.js")],
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
    )
    assert result.returncode == 0
    assert "STALE_UNSUPPORTED_DIST_MAC" in result.stdout
    assert "Duplicate Spotlight risk" in result.stderr
