from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import doctor  # type: ignore  # noqa: E402


def test_supported_python_version_range() -> None:
    assert doctor.MIN_VERSION <= (3, 12) <= doctor.MAX_VERSION
    assert doctor.MIN_VERSION <= (3, 11) <= doctor.MAX_VERSION
    assert doctor.MIN_VERSION <= (3, 10) <= doctor.MAX_VERSION
    # 3.10 is still in range but the doctor warns about it
    assert not (doctor.MIN_VERSION <= (3, 14) <= doctor.MAX_VERSION)
    assert not (doctor.MIN_VERSION <= (3, 9) <= doctor.MAX_VERSION)


def test_existing_bad_venv_status_is_detected(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "venv_python", lambda: Path("/tmp/fake-python"))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(doctor, "_probe_python", lambda cmd: doctor.CheckResult(False, version=(3, 14, 0), executable=cmd[0], error="unsupported Python 3.14"))
    status = doctor.existing_venv_status()
    assert status is not None
    assert status.ok is False
    assert "3.14" in (status.error or "")
