"""Electron packaging and helper script configuration checks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = json.loads((ROOT / "electron-app" / "package.json").read_text())
README = (ROOT / "README.md").read_text()


def test_package_builds_supported_desktop_targets() -> None:
    scripts = PACKAGE["scripts"]
    assert "--arm64" in scripts["build:mac:arm64:raw"]
    assert "--arm64" in scripts["build:win:arm64:raw"]
    assert "--arm64" in scripts["build:linux:arm64:raw"]
    assert "verify:resources:native" in scripts["build:mac:arm64"]
    assert "verify:resources:native" in scripts["build:win:arm64"]
    assert "verify:resources:native" in scripts["build:linux:arm64"]
    for name in ("build:mac:x64", "build:mac:universal"):
        assert "unsupported-arch.js" in scripts[name]
    assert "build:win:x64:raw" in scripts["build:win:x64"]
    assert "build:linux:x64:raw" in scripts["build:linux:x64"]
    assert PACKAGE["build"]["mac"]["target"][0]["arch"] == ["arm64"]
    assert PACKAGE["build"]["win"]["target"][0]["arch"] == ["arm64", "x64"]
    assert PACKAGE["build"]["linux"]["target"][0]["arch"] == ["arm64", "x64"]


def test_readme_platform_table_matches_package_scripts() -> None:
    assert "macOS Apple Silicon only" in README
    assert "Windows arm64 and x64" in README
    assert "Linux arm64 and x64" in README
    assert "Intel Mac / macOS x64" in README
    assert "npm run build:mac" in README
    assert "dist/mac-arm64/DepthLens Pro.app" in README
    assert "verify-packaged-resources.js" in README


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


def test_electron_navigation_policy_rejects_other_localhost_ports() -> None:
    policy = (ROOT / "electron-app" / "src" / "security-policy.js").read_text()
    test_policy = (ROOT / "electron-app" / "test-security-policy.js").read_text()
    assert "parsedUrl.hostname === backendHost" in policy
    assert 'String(parsedUrl.port || "80") === String(backendPort)' in policy
    assert '"http://127.0.0.1:8766/live"' in test_policy
    assert '"http://localhost:8765/live"' in test_policy


def test_backend_lifecycle_ignores_empty_candidate_paths() -> None:
    policy = (ROOT / "electron-app" / "src" / "backend-process-policy.js").read_text()
    test_policy = (ROOT / "electron-app" / "test-security-policy.js").read_text()
    assert "if (!trimmed) return null" in policy
    assert 'buildOwnershipCandidatePaths({ cwd: "", backendDir: "" })' in test_policy


def test_backend_lifecycle_rejects_unrelated_uvicorn_processes() -> None:
    test_policy = (ROOT / "electron-app" / "test-security-policy.js").read_text()
    assert "python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765" in test_policy
    assert "false" in test_policy


def test_backend_spawn_keeps_shell_disabled_and_args_array() -> None:
    lifecycle = (ROOT / "electron-app" / "src" / "main" / "backend-lifecycle.js").read_text()
    main = (ROOT / "electron-app" / "main.js").read_text()

    assert 'require("./src/main/backend-lifecycle")' in main
    assert "createBackendLifecycle({" in main
    assert "backendLifecycle.startBackend()" in main
    assert 'const args = ["-m", "uvicorn", "backend.app:app"' in lifecycle
    assert "spawn(pythonPath, args" in lifecycle
    assert "shell: false" in lifecycle


def test_packaged_startup_requires_models_and_onnx() -> None:
    lifecycle = (ROOT / "electron-app" / "src" / "main" / "backend-lifecycle.js").read_text()
    main = (ROOT / "electron-app" / "main.js").read_text()

    assert 'require("./src/main/backend-lifecycle")' in main
    assert "backendLifecycle.startBackend()" in main
    assert '["models/", details.modelsDir]' in lifecycle
    assert '["models/onnx/", details.onnxDir]' in lifecycle
    assert "stale installed copy" in lifecycle
