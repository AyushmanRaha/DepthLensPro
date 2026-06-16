from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import doctor  # type: ignore  # noqa: E402
import platform_support  # type: ignore  # noqa: E402


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
    monkeypatch.setattr(
        doctor,
        "_probe_python",
        lambda cmd: doctor.CheckResult(
            False, version=(3, 14, 0), executable=cmd[0], error="unsupported Python 3.14"
        ),
    )
    status = doctor.existing_venv_status()
    assert status is not None
    assert status.ok is False
    assert "3.14" in (status.error or "")


def test_onnx_model_list_parsing() -> None:
    assert doctor.parse_onnx_model_list("midas_small") == ["midas_small"]
    assert doctor.parse_onnx_model_list("midas_small,dpt_hybrid") == ["midas_small", "dpt_hybrid"]
    assert doctor.parse_onnx_model_list("all") == ["midas_small", "dpt_hybrid", "dpt_large"]


def test_doctor_argument_parsing_cross_platform(monkeypatch) -> None:
    for system, machine in [
        ("Darwin", "arm64"),
        ("Windows", "ARM64"),
        ("Windows", "AMD64"),
        ("Linux", "aarch64"),
        ("Linux", "x86_64"),
    ]:
        monkeypatch.setattr(doctor.platform, "system", lambda s=system: s)
        monkeypatch.setattr(doctor.platform, "machine", lambda m=machine: m)
        args = doctor.parse_args(
            ["--with-onnx", "--onnx-models", "midas_small,dpt_hybrid", "--onnx-strict"]
        )
        assert args.with_onnx is True
        assert doctor.parse_onnx_model_list(args.onnx_models) == ["midas_small", "dpt_hybrid"]
        assert platform_support.evaluate_target(system, machine).supported


def test_platform_support_matrix() -> None:
    assert platform_support.evaluate_target("Darwin", "arm64").supported
    assert not platform_support.evaluate_target("Darwin", "x64").supported
    assert platform_support.evaluate_target("Windows", "AMD64").supported
    assert platform_support.evaluate_target("Linux", "x86_64").supported


def test_default_setup_exports_onnx() -> None:
    args = doctor.parse_args([])
    assert doctor.should_export_onnx(args, stdin_is_tty=False) is True
    args = doctor.parse_args(["--without-onnx"])
    assert doctor.should_export_onnx(args, stdin_is_tty=False) is False


def test_onnx_export_command_models() -> None:
    args = doctor.parse_args(
        ["--with-onnx", "--onnx-models", "midas_small,dpt_hybrid", "--onnx-force"]
    )
    cmd = doctor.onnx_export_command(Path("python"), args)
    assert "--models" in cmd
    assert "midas_small,dpt_hybrid" in cmd
    assert "--force" in cmd
