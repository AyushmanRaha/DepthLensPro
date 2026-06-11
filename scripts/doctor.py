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
SUPPORTED_ARCHES = {"arm64", "aarch64"}


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
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


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
        for minor in (12, 11, 10):
            candidates.append(Candidate([f"/Library/Frameworks/Python.framework/Versions/3.{minor}/bin/python3"], f"python.org 3.{minor}"))
        for minor in (12, 11, 10):
            candidates.append(Candidate([f"/opt/homebrew/bin/python3.{minor}"], f"Homebrew Apple Silicon 3.{minor}"))
            candidates.append(Candidate([f"/usr/local/bin/python3.{minor}"], f"Homebrew Intel/Rosetta 3.{minor}"))
        candidates.append(Candidate(["python3"], "PATH python3"))
    elif system == "Windows":
        for minor in (12, 11, 10):
            candidates.append(Candidate(["py", f"-3.{minor}"], f"Windows py launcher 3.{minor}"))
        candidates.append(Candidate(["python"], "PATH python"))
    else:
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
            return cand.command, result, failures
        failures.append((cand, result))
    lines = ["No working supported Python found (required: 3.10-3.12 with ensurepip, ssl, venv, pyexpat)."]
    for cand, res in failures:
        lines.append(f"- {cand.label} ({' '.join(cand.command)}): {res.error}")
    if platform.system() == "Darwin":
        lines.append("Recommended macOS remediation: install Python 3.12 from https://www.python.org/downloads/macos/ if Homebrew Python fails ensurepip/pyexpat.")
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
    if args.enforce_arch and arch not in SUPPORTED_ARCHES:
        raise SystemExit(f"Unsupported native app architecture {platform.machine()}. Supported native builds are ARM64/aarch64.")
    selected_cmd, selected, failures = find_python()
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
    return p.parse_args(argv)


if __name__ == "__main__":
    setup(parse_args())
