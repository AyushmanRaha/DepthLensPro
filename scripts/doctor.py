#!/usr/bin/env python3
"""Cross-platform setup/doctor for DepthLens Pro.

This script intentionally runs before the project virtualenv exists.  It finds a
supported Python (3.10-3.12), verifies core stdlib modules that frequently break
on misconfigured installs, creates/repairs the repo-root venv, installs backend
and Electron dependencies, and prints a deterministic summary.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "venv"
MIN_VERSION = (3, 10)
MAX_VERSION = (3, 12)
REQUIRED_STDLIB = ("ensurepip", "ssl", "venv", "pyexpat")
PYTHON_310_WARNING = (
    "Python 3.10 was selected. The pinned backend dependencies are only guaranteed "
    "on Python 3.11-3.12. If pip install fails, install Python 3.12 from "
    "python.org and re-run."
)
SUPPORTED_ARCHES = {"arm64", "aarch64", "x86_64", "amd64"}
ONNX_MODEL_IDS = ("midas_small", "dpt_hybrid", "dpt_large")
DETECTOR_TORCH_CACHE = ROOT / "models" / "torch-cache"


def parse_onnx_model_list(value: str | None) -> list[str]:
    """Parse a comma-separated ONNX model selection into canonical model IDs."""
    if value is None or value == "":
        return ["midas_small"]
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if raw == ["all"]:
        return list(ONNX_MODEL_IDS)
    invalid = [item for item in raw if item not in ONNX_MODEL_IDS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported ONNX model(s): {', '.join(invalid)}. Expected one of: {', '.join(ONNX_MODEL_IDS)} or all")
    return raw


def verify_detector_cache(cache_root: Path = DETECTOR_TORCH_CACHE) -> tuple[bool, str]:
    """Verify TorchVision detector checkpoints exist in the setup-created cache."""
    checkpoints = cache_root / "hub" / "checkpoints"
    if not checkpoints.is_dir():
        return False, f"Detector checkpoint directory is missing: {checkpoints}"
    if not any(path.is_file() and path.suffix == ".pth" for path in checkpoints.iterdir()):
        return False, f"No .pth detector checkpoint files found in: {checkpoints}"
    return True, f"Detector weights cached under {cache_root.relative_to(ROOT) if cache_root.is_relative_to(ROOT) else cache_root}"


def should_prefetch_detector_weights(args: argparse.Namespace) -> bool:
    """Return whether normal setup should cache RGB detector weights."""
    if args.with_detector_weights and args.without_detector_weights:
        raise SystemExit("Choose either --with-detector-weights or --without-detector-weights, not both.")
    if args.without_detector_weights:
        return False
    if args.doctor_only:
        return False
    if os.environ.get("CI") == "1" or os.environ.get("TESTING") == "1":
        return bool(args.with_detector_weights)
    return True


def prefetch_detector_weights(py: Path, env: dict[str, str]) -> None:
    print("Caching RGB object detector weights for offline / packaged use...")
    detector_env = env.copy()
    detector_env.pop("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", None)
    detector_env["TORCH_HOME"] = str(DETECTOR_TORCH_CACHE)
    _run([str(py), str(ROOT / "scripts" / "prefetch-detector-weights.py")], env=detector_env)
    ok, message = verify_detector_cache()
    if not ok:
        raise SystemExit(message)
    print(message)


def should_export_onnx(args: argparse.Namespace, *, stdin_is_tty: bool | None = None) -> bool:
    """Return whether setup should export/validate ONNX. Default is non-interactive No."""
    if args.with_onnx and args.without_onnx:
        raise SystemExit("Choose either --with-onnx or --without-onnx, not both.")
    if args.with_onnx or args.onnx_validate_only:
        return True
    if args.without_onnx:
        return False
    if stdin_is_tty is None:
        stdin_is_tty = sys.stdin.isatty()
    if not stdin_is_tty:
        print("ONNX export skipped by default in non-interactive setup. Pass --with-onnx to export or --without-onnx to make the skip explicit.")
        return False
    answer = input("Export optional ONNX model files now? This may download large weights. [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def onnx_export_command(py: Path, args: argparse.Namespace) -> list[str]:
    models = parse_onnx_model_list(args.onnx_models)
    cmd = [str(py), str(ROOT / "backend" / "scripts" / "export_onnx.py"), "--output-dir", str(ROOT / "models" / "onnx")]
    if args.onnx_validate_only:
        cmd.append("--validate-only")
    if args.onnx_force:
        cmd.append("--force")
    if args.onnx_strict:
        cmd.append("--strict")
    if models == list(ONNX_MODEL_IDS):
        cmd.append("--all")
    elif len(models) == 1:
        cmd.extend(["--model", models[0]])
    else:
        cmd.extend(["--models", ",".join(models)])
    return cmd


@dataclass
class Candidate:
    command: list[str]
    label: str


@dataclass
class CheckResult:
    ok: bool
    version: tuple[int, int, int] | None = None
    executable: str | None = None
    error: str | None = None


def _run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="" if exc.stdout.endswith("\n") else "\n")
        raise


def _probe_python(command: list[str]) -> CheckResult:
    code = """
import importlib, json, sys
mods = ['ensurepip', 'ssl', 'venv', 'pyexpat']
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f'{mod}: {type(exc).__name__}: {exc}')
try:
    import ensurepip
    ensurepip.version()
except Exception as exc:
    missing.append(f'ensurepip.version: {type(exc).__name__}: {exc}')
print(json.dumps({'version': list(sys.version_info[:3]), 'executable': sys.executable, 'missing': missing}))
raise SystemExit(1 if missing else 0)
"""
    try:
        proc = subprocess.run(command + ["-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(False, error=f"{type(exc).__name__}: {exc}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        version = tuple(int(x) for x in payload["version"])
        executable = str(payload["executable"])
        missing = payload.get("missing") or []
    except Exception:
        return CheckResult(False, error=proc.stdout.strip() or "could not run Python")
    if not (MIN_VERSION <= version[:2] <= MAX_VERSION):
        return CheckResult(False, version=version, executable=executable, error=f"unsupported Python {version[0]}.{version[1]}; DepthLens Pro supports 3.10-3.12")
    if proc.returncode != 0 or missing:
        return CheckResult(False, version=version, executable=executable, error="; ".join(missing) or proc.stdout.strip())
    return CheckResult(True, version=version, executable=executable)


def candidate_pythons() -> list[Candidate]:
    system = platform.system()
    candidates: list[Candidate] = []
    if system == "Darwin":
        for minor in (12, 11):
            candidates.append(Candidate([f"/Library/Frameworks/Python.framework/Versions/3.{minor}/bin/python3"], f"python.org 3.{minor}"))
        for minor in (12, 11):
            candidates.append(Candidate([f"/opt/homebrew/bin/python3.{minor}"], f"Homebrew Apple Silicon 3.{minor}"))
            candidates.append(Candidate([f"/usr/local/bin/python3.{minor}"], f"Homebrew Intel/Rosetta 3.{minor}"))
        candidates.append(Candidate(["/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"], "python.org 3.10"))
        candidates.append(Candidate(["/opt/homebrew/bin/python3.10"], "Homebrew Apple Silicon 3.10"))
        candidates.append(Candidate(["/usr/local/bin/python3.10"], "Homebrew Intel/Rosetta 3.10"))
        candidates.append(Candidate(["python3"], "PATH python3"))
    elif system == "Windows":
        for minor in (12, 11, 10):
            candidates.append(Candidate(["py", f"-3.{minor}"], f"Windows py launcher 3.{minor}"))
        candidates.append(Candidate(["python"], "PATH python"))
    else:
        for minor in (12, 11):
            candidates.append(Candidate([f"/usr/bin/python3.{minor}"], f"system python3.{minor}"))
        for minor in (12, 11, 10):
            candidates.append(Candidate([f"python3.{minor}"], f"PATH python3.{minor}"))
        candidates.append(Candidate(["python3"], "PATH python3"))
    return candidates


def find_python() -> tuple[list[str], CheckResult, list[tuple[Candidate, CheckResult]]]:
    failures: list[tuple[Candidate, CheckResult]] = []
    seen: set[tuple[str, ...]] = set()
    for cand in candidate_pythons():
        if tuple(cand.command) in seen:
            continue
        seen.add(tuple(cand.command))
        exe = cand.command[0]
        if (os.path.sep in exe or (os.path.altsep and os.path.altsep in exe)) and not Path(exe).exists():
            continue
        if os.path.sep not in exe and shutil.which(exe) is None:
            continue
        result = _probe_python(cand.command)
        if result.ok:
            print(f"Selected Python: {result.executable or ' '.join(cand.command)} (version {'.'.join(map(str, result.version or ()))})")
            return cand.command, result, failures
        failures.append((cand, result))
    lines = ["No working supported Python found (required: 3.10-3.12 with ensurepip, ssl, venv, pyexpat)."]
    for cand, res in failures:
        lines.append(f"- {cand.label} ({' '.join(cand.command)}): {res.error}")
    if platform.system() == "Darwin":
        lines.append("Recommended macOS remediation: install Python 3.12 from https://www.python.org/downloads/macos/ if Homebrew Python fails ensurepip/pyexpat. " + PYTHON_310_WARNING)
    elif platform.system() == "Windows":
        lines.append("Recommended Windows remediation: install Python 3.12 for ARM64/x64 from python.org and enable the py launcher.")
    else:
        lines.append("Recommended Linux remediation: install python3.12/python3.11 plus the matching venv package (for example python3.12-venv).")
    raise SystemExit("\n".join(lines))


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def existing_venv_status() -> CheckResult | None:
    py = venv_python()
    if not py.exists():
        return None
    return _probe_python([str(py)])


def recreate_venv(python_cmd: list[str]) -> None:
    if VENV.exists():
        print(f"Removing unsupported/broken venv at {VENV}")
        shutil.rmtree(VENV)
    _run(python_cmd + ["-m", "venv", str(VENV)])
    status = existing_venv_status()
    if not status or not status.ok:
        raise SystemExit(f"Created venv is not usable: {status.error if status else 'missing python'}")


def cert_env(py: Path) -> dict[str, str]:
    env = os.environ.copy()
    code = "import certifi; print(certifi.where())"
    proc = subprocess.run([str(py), "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.returncode == 0:
        bundle = proc.stdout.strip()
        env["SSL_CERT_FILE"] = bundle
        env["REQUESTS_CA_BUNDLE"] = bundle
        env["CURL_CA_BUNDLE"] = bundle
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("DEPTHLENS_DISABLE_MODEL_DOWNLOADS", "1")
    env.setdefault("DEPTHLENS_SKIP_WARMUP", "1")
    env.setdefault("DEPTHLENS_CACHE_BACKEND", "memory")
    return env


def setup(args: argparse.Namespace) -> dict[str, str]:
    arch = platform.machine().lower()
    system = platform.system()
    if args.enforce_arch:
        if system == "Darwin" and arch not in {"arm64", "aarch64"}:
            raise SystemExit(f"Unsupported macOS native app architecture {platform.machine()}. Supported macOS native builds are Apple Silicon arm64 only.")
        if system in {"Windows", "Linux"} and arch not in SUPPORTED_ARCHES:
            raise SystemExit(f"Unsupported native app architecture {platform.machine()}. Supported Windows/Linux native builds are arm64/aarch64 and x64/x86_64.")
    selected_cmd, selected, failures = find_python()
    if selected.version and selected.version[:2] == (3, 10):
        print(f"WARNING: {PYTHON_310_WARNING}")
    if platform.system() == "Darwin" and "zsh" in os.environ.get("SHELL", ""):
        print("Tip: if pasting multi-line command blocks into Terminal causes 'zsh: command not found: #', run: setopt interactivecomments")
    status = existing_venv_status()
    if status is None or not status.ok:
        recreate_venv(selected_cmd)
    else:
        print(f"Existing venv is valid: Python {'.'.join(map(str, status.version or ())) } at {status.executable}")
    py = venv_python()
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "certifi"])
    env = cert_env(py)
    if not args.doctor_only:
        _run([str(py), "-m", "pip", "install", "-r", str(ROOT / "backend" / "requirements.txt")], env=env)
    pip_check = _run([str(py), "-m", "pip", "check"], check=False, env=env)
    if pip_check.returncode != 0:
        raise SystemExit(pip_check.stdout)
    if not args.doctor_only:
        _run(["npm", "install"], cwd=ROOT / "electron-app")
    DETECTOR_TORCH_CACHE.mkdir(parents=True, exist_ok=True)
    (ROOT / "models" / "onnx").mkdir(parents=True, exist_ok=True)
    for keep in [ROOT / "models" / ".gitkeep", ROOT / "models" / "onnx" / ".gitkeep"]:
        if not keep.exists():
            keep.touch()
    print("Ensured models/onnx and models/torch-cache directory structure.")
    if should_prefetch_detector_weights(args):
        try:
            prefetch_detector_weights(py, env)
        except subprocess.CalledProcessError as exc:
            raise SystemExit("RGB detector weights could not be cached. Re-run setup with network access, or pass --without-detector-weights to skip RGB object detection support.") from exc
    else:
        print("WARNING: RGB Camera detection may fail until detector weights are cached.")
    export_onnx = should_export_onnx(args)
    if export_onnx:
        _run(onnx_export_command(py, args), env=env)
    else:
        print("ONNX export intentionally skipped; PyTorch fallback remains available and packaged ONNX verification will be optional.")
    resources = _run(["npm", "run", "verify:resources"], cwd=ROOT / "electron-app", check=False)
    node_v = _run(["node", "--version"], check=False).stdout.strip() if shutil.which("node") else "missing"
    npm_v = _run(["npm", "--version"], check=False).stdout.strip() if shutil.which("npm") else "missing"
    onnx = _run([str(py), "-c", "from backend.services.onnx_diagnostics import onnx_status_payload; import json; print(json.dumps(onnx_status_payload(), default=str))"], check=False, env={**env, "PYTHONPATH": str(ROOT)})
    summary = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "selected_python": selected.executable or " ".join(selected_cmd),
        "python_version": ".".join(map(str, selected.version or ())),
        "venv_python": str(py),
        "pip_version": _run([str(py), "-m", "pip", "--version"], check=False).stdout.strip(),
        "node_version": node_v,
        "npm_version": npm_v,
        "backend_dependency_status": "ok",
        "resource_verification_status": "ok" if resources.returncode == 0 else "failed",
        "onnx_status_summary": "ok" if onnx.returncode == 0 else "unavailable/degraded",
    }
    print("\nDepthLens Pro environment summary")
    print(json.dumps(summary, indent=2))
    if resources.returncode != 0:
        raise SystemExit(resources.stdout)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DepthLens Pro setup and environment doctor")
    p.add_argument("--doctor-only", action="store_true", help="check/create venv and verify tools without installing app dependencies")
    p.add_argument("--enforce-arch", action="store_true", help="fail if the current machine cannot build the supported native app")
    p.add_argument("--with-onnx", action="store_true", help="export/validate requested ONNX models during setup")
    p.add_argument("--with-detector-weights", action="store_true", help="cache RGB object detector weights during setup even when CI/TESTING is set")
    p.add_argument("--without-detector-weights", action="store_true", help="skip RGB object detector weight caching")
    p.add_argument("--without-onnx", action="store_true", help="skip ONNX export explicitly")
    p.add_argument("--onnx-models", default="midas_small", help="comma-separated ONNX models: midas_small, dpt_hybrid, dpt_large, or all")
    p.add_argument("--onnx-strict", action="store_true", help="fail setup if any requested ONNX model is missing or invalid")
    p.add_argument("--onnx-validate-only", action="store_true", help="validate requested ONNX models without exporting")
    p.add_argument("--onnx-force", action="store_true", help="regenerate requested ONNX models even when cached files validate")
    args = p.parse_args(argv)
    parse_onnx_model_list(args.onnx_models)
    if args.with_onnx and args.without_onnx:
        p.error("choose either --with-onnx or --without-onnx, not both")
    if args.with_detector_weights and args.without_detector_weights:
        p.error("choose either --with-detector-weights or --without-detector-weights, not both")
    return args


if __name__ == "__main__":
    setup(parse_args())
