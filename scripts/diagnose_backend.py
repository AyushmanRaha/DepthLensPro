#!/usr/bin/env python3
"""Cross-platform DepthLens backend diagnostics for macOS ARM, Windows ARM, and Linux ARM."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("DEPTHLENS_BACKEND_PORT", "8765"))
HOST = "127.0.0.1"
MODELS = ("midas_small", "dpt_hybrid", "dpt_large")


def run(cmd: list[str], timeout: int = 4) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout.strip()
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def version(cmd: list[str]) -> str:
    code, out = run(cmd)
    return out.splitlines()[-1] if code == 0 and out else "missing/unavailable"


def venv_python() -> Path:
    return (
        ROOT / "venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    )


def http_json(path: str, timeout: float = 2.0) -> dict[str, object]:
    url = f"http://{HOST}:{PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body) if body else None
            except json.JSONDecodeError:
                payload = body[:500]
            return {
                "ok": 200 <= response.status < 400,
                "status": response.status,
                "url": url,
                "body": payload,
            }
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def socket_port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, PORT)) == 0


def port_owner() -> dict[str, object]:
    system = platform.system()
    if system == "Windows":
        if shutil.which("powershell.exe"):
            code, out = run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    f'$c=Get-NetTCPConnection -LocalPort {PORT} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) {{ $p=Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)"; [pscustomobject]@{{Pid=$c.OwningProcess;CommandLine=$p.CommandLine}} | ConvertTo-Json -Compress }}',
                ]
            )
            if code == 0 and out:
                try:
                    data = json.loads(out)
                    return {
                        "pid": data.get("Pid"),
                        "command_line": data.get("CommandLine"),
                        "method": "Get-NetTCPConnection",
                    }
                except json.JSONDecodeError:
                    pass
        code, out = run(["cmd.exe", "/c", f"netstat -ano -p tcp | findstr :{PORT}"])
        for line in out.splitlines():
            if "LISTENING" in line:
                pid = line.split()[-1]
                _, cmd = run(
                    ["cmd.exe", "/c", f"wmic process where processid={pid} get CommandLine /value"]
                )
                return {"pid": pid, "command_line": cmd, "method": "netstat/tasklist"}
    else:
        if shutil.which("lsof"):
            code, out = run(["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN"])
            for line in out.splitlines():
                if "LISTEN" in line and not line.startswith("COMMAND"):
                    parts = line.split()
                    pid = parts[1] if len(parts) > 1 else None
                    _, cmd = run(["ps", "-p", str(pid), "-o", "command="]) if pid else (1, "")
                    return {"pid": pid, "command_line": cmd, "method": "lsof/ps"}
        if shutil.which("ss"):
            code, out = run(["ss", "-ltnp", f"sport = :{PORT}"])
            return {
                "pid": None,
                "command_line": out,
                "method": "ss",
                "warning": "PID discovery may require elevated permissions.",
            }
    return {
        "pid": None,
        "command_line": None,
        "method": "python-socket",
        "warning": "PID discovery unavailable; continuing diagnostics.",
    }


def electron_log_path() -> str:
    home = Path.home()
    if platform.system() == "Darwin":
        return str(home / "Library/Logs/DepthLens Pro/main.log")
    if platform.system() == "Windows":
        return str(
            Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
            / "DepthLens Pro/logs/main.log"
        )
    return str(
        Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "DepthLens Pro/logs/main.log"
    )


def remediations() -> list[str]:
    if platform.system() == "Windows":
        return [
            r".\scripts\setup-windows.ps1 --without-onnx",
            r".\scripts\build-native-windows.ps1 --without-onnx",
            f"taskkill /PID <pid> /F  # only if PID owns stale DepthLens on {PORT}",
        ]
    if platform.system() == "Darwin":
        return [
            "scripts/setup-macos.sh --without-onnx",
            "scripts/build-native-macos.sh --without-onnx",
            f"kill <pid>  # only if PID owns stale DepthLens on {PORT}",
            "Replace stale /Applications/DepthLens Pro.app only after building a fresh app.",
        ]
    return [
        "scripts/setup-linux.sh --without-onnx",
        "scripts/build-native-linux.sh --without-onnx",
        f"kill <pid>  # only if PID owns stale DepthLens on {PORT}",
    ]


def main() -> int:
    vp = venv_python()
    model_dir = ROOT / "models"
    onnx_dir = Path(
        os.environ.get("DEPTHLENS_ONNX_DIR")
        or os.environ.get("ONNX_WEIGHTS_DIR")
        or model_dir / "onnx"
    )
    occupied = socket_port_open()
    report = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "repo_root": str(ROOT),
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "venv_python_path": str(vp),
        "venv_python_version": version([str(vp), "--version"]) if vp.exists() else "missing",
        "node_version": version(["node", "--version"]),
        "npm_version": version(["npm", "--version"]),
        "backend_port": PORT,
        "port_8765_occupied": occupied,
        "port_owner": (
            port_owner()
            if occupied
            else {"pid": None, "command_line": None, "method": "python-socket", "status": "free"}
        ),
        "live": http_json("/live"),
        "ready": http_json("/ready", timeout=5.0),
        "onnx_status": http_json("/onnx/status", timeout=5.0),
        "model_directory": str(model_dir),
        "onnx_directory": str(onnx_dir),
        "onnx_files": {
            m: {
                "path": str(onnx_dir / f"{m}.onnx"),
                "exists": (onnx_dir / f"{m}.onnx").is_file(),
                "size": (
                    (onnx_dir / f"{m}.onnx").stat().st_size
                    if (onnx_dir / f"{m}.onnx").is_file()
                    else 0
                ),
            }
            for m in MODELS
        },
        "electron_log_path": electron_log_path(),
        "suggested_remediation_commands": remediations(),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
