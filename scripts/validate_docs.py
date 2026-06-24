#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SCRIPT_RE = re.compile(r"npm\s+--prefix\s+electron-app\s+run\s+([\w:.-]+)")
STALE = ("docs/engineering-audit", "docs/refactor-plan", "docs/old-", "docs/archive/")


def load_scripts():
    pkg = json.loads((ROOT / "electron-app/package.json").read_text())
    return set(pkg.get("scripts", {}))


def route_paths():
    txt = (ROOT / "backend/api/routes.py").read_text()
    return set(re.findall(r'@router\.(?:get|post|delete|put|patch)\("([^"]+)"', txt))


def local_target_exists(src: Path, target: str) -> bool:
    target = target.split("#", 1)[0]
    if not target or re.match(r"^[a-z]+:", target) or target.startswith("mailto:"):
        return True
    if target.startswith("/"):
        return True
    return (src.parent / target).resolve().exists()


def main():
    errors = []
    scripts = load_scripts()
    routes = route_paths()
    docs = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("**/*.md"))]
    for path in docs:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for stale in STALE:
            if stale in text:
                errors.append(f"{rel}: links stale engineering/refactor doc {stale}")
        for m in MD_LINK.finditer(text):
            url = m.group(1).strip()
            if not local_target_exists(path, url):
                errors.append(f"{rel}: missing local link/image {url}")
        for m in SCRIPT_RE.finditer(text):
            script = (m.group(1) or "").strip()
            if script and script not in scripts:
                errors.append(f"{rel}: referenced npm script does not exist: {script}")
    api_ref = ROOT / "docs/api-reference.md"
    if api_ref.exists():
        text = api_ref.read_text(encoding="utf-8")
        for endpoint in sorted(set(re.findall(r"`((?:/api)?/[a-z0-9_/{}/.-]+)`", text))):
            if (
                "{" in endpoint
                or endpoint.startswith("/live")
                or endpoint.startswith("/ready")
                or endpoint.startswith("/health")
            ):
                continue
            if endpoint not in routes:
                errors.append(
                    f"docs/api-reference.md: documented endpoint not declared: {endpoint}"
                )
    if (ROOT / "package-lock.json").exists() and (ROOT / "package-lock.json").is_file():
        # untracked local files are common in Codex; only fail if Git tracks it.
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "package-lock.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode == 0:
            errors.append(
                "root package-lock.json is tracked; use electron-app/package-lock.json only"
            )
    if errors:
        for e in errors:
            print(f"::error::{e}")
        return 1
    print("Documentation contract passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
