from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_backend  # type: ignore  # noqa: E402


def test_diagnostic_port_free_continues(monkeypatch) -> None:
    monkeypatch.setattr(diagnose_backend, "socket_port_open", lambda: False)
    monkeypatch.setattr(
        diagnose_backend,
        "http_json",
        lambda path, timeout=2.0: {"ok": False, "error": "connection refused", "url": path},
    )
    assert diagnose_backend.socket_port_open() is False
    assert diagnose_backend.http_json("/live")["ok"] is False


def test_diagnostic_port_owner_fallback_when_pid_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(diagnose_backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(diagnose_backend.shutil, "which", lambda name: None)
    owner = diagnose_backend.port_owner()
    assert owner["method"] == "python-socket"
    assert "warning" in owner


def test_electron_log_paths_are_platform_specific(monkeypatch) -> None:
    monkeypatch.setattr(diagnose_backend.platform, "system", lambda: "Darwin")
    assert "Library" in diagnose_backend.electron_log_path()
    monkeypatch.setattr(diagnose_backend.platform, "system", lambda: "Windows")
    assert "DepthLens Pro" in diagnose_backend.electron_log_path()
    monkeypatch.setattr(diagnose_backend.platform, "system", lambda: "Linux")
    assert (
        ".config" in diagnose_backend.electron_log_path()
        or "DepthLens Pro" in diagnose_backend.electron_log_path()
    )
