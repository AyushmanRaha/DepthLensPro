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
    for system, machine in [("Darwin", "arm64"), ("Windows", "ARM64"), ("Linux", "aarch64")]:
        monkeypatch.setattr(doctor.platform, "system", lambda s=system: s)
        monkeypatch.setattr(doctor.platform, "machine", lambda m=machine: m)
        args = doctor.parse_args(
            ["--with-onnx", "--onnx-models", "midas_small,dpt_hybrid", "--onnx-strict"]
        )
        assert args.with_onnx is True
        assert doctor.parse_onnx_model_list(args.onnx_models) == ["midas_small", "dpt_hybrid"]
        assert doctor.SUPPORTED_ARCHES & {doctor.platform.machine().lower()}


def test_noninteractive_default_skips_onnx() -> None:
    args = doctor.parse_args([])
    assert doctor.should_export_onnx(args, stdin_is_tty=False) is False


def test_interactive_prompt_default_no_skips_onnx(monkeypatch) -> None:
    args = doctor.parse_args([])
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert doctor.should_export_onnx(args, stdin_is_tty=True) is False


def test_onnx_export_command_models() -> None:
    args = doctor.parse_args(
        ["--with-onnx", "--onnx-models", "midas_small,dpt_hybrid", "--onnx-force"]
    )
    cmd = doctor.onnx_export_command(Path("python"), args)
    assert "--models" in cmd
    assert "midas_small,dpt_hybrid" in cmd
    assert "--force" in cmd


def test_detector_cache_verification_fails_when_checkpoints_missing(tmp_path) -> None:
    ok, message = doctor.verify_detector_cache(tmp_path / "torch-cache")
    assert ok is False
    assert "missing" in message


def test_detector_cache_verification_fails_without_pth(tmp_path) -> None:
    checkpoints = tmp_path / "torch-cache" / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "README.txt").write_text("not weights")
    ok, message = doctor.verify_detector_cache(tmp_path / "torch-cache")
    assert ok is False
    assert "No .pth" in message


def test_detector_cache_verification_passes_with_pth(tmp_path) -> None:
    checkpoints = tmp_path / "torch-cache" / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "fake-detector.pth").write_bytes(b"fake")
    ok, message = doctor.verify_detector_cache(tmp_path / "torch-cache")
    assert ok is True
    assert "Detector weights cached" in message


def test_detector_weight_argument_parsing() -> None:
    with_weights = doctor.parse_args(["--with-detector-weights"])
    assert with_weights.with_detector_weights is True
    assert with_weights.without_detector_weights is False

    without_weights = doctor.parse_args(["--without-detector-weights"])
    assert without_weights.with_detector_weights is False
    assert without_weights.without_detector_weights is True


def test_detector_weight_argument_conflict_errors() -> None:
    try:
        doctor.parse_args(["--with-detector-weights", "--without-detector-weights"])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover
        raise AssertionError("conflicting detector weight flags should error")


def test_onnx_verify_mode_mapping() -> None:
    assert doctor.onnx_verify_mode(doctor.parse_args([])) == "optional"
    assert doctor.onnx_verify_mode(doctor.parse_args(["--without-onnx"])) == "off"
    assert (
        doctor.onnx_verify_mode(doctor.parse_args(["--with-onnx", "--onnx-models", "midas_small"]))
        == "required"
    )
    assert (
        doctor.onnx_verify_mode(doctor.parse_args(["--with-onnx", "--onnx-models", "all"]))
        == "require-all"
    )
    assert (
        doctor.onnx_verify_mode(doctor.parse_args(["--onnx-validate-only", "--onnx-models", "all"]))
        == "require-all"
    )


def test_step_helpers_are_available() -> None:
    for name in [
        "select_python",
        "ensure_venv",
        "install_python_dependencies",
        "install_electron_dependencies",
        "ensure_model_dirs",
        "cache_midas_assets",
        "cache_detector_assets",
        "handle_onnx_assets",
        "verify_resources",
    ]:
        assert callable(getattr(doctor, name))


def test_onnx_validate_only_command_never_forces_export() -> None:
    args = doctor.parse_args(["--onnx-validate-only", "--onnx-models", "all"])
    cmd = doctor.onnx_export_command(Path("python"), args)
    assert "--validate-only" in cmd
    assert "--force" not in cmd
