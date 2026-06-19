#!/usr/bin/env python3
"""Detect and classify changed files for local and GitHub Actions CI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

DOC_EXTS = {".md", ".markdown", ".rst", ".txt"}
DOC_FILES = {"README.md", "LICENSE", "SECURITY.md", "CONTRIBUTING.md"}
DOC_DIRS = ("docs/",)
DOC_ASSET_DIRS = ("docs/screenshots/", "docs/assets/", "images/", "assets/docs/")
WORKFLOW_FILES = {"scripts/ci.sh", "scripts/validate_workflows.py", "scripts/changed_files.py"}


def run_git(args: list[str], *, check: bool = False) -> str:
    proc = subprocess.run(
        ["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def maybe_fetch(ref: str) -> None:
    if not ref or ref_exists(ref):
        return
    name = ref.removeprefix("origin/")
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=100", "origin", f"{name}:{name}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=100", "origin", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def changed_between(base: str | None, head: str | None) -> list[str]:
    head = head or "HEAD"
    if not base:
        parent = run_git(["rev-list", "--parents", "-n", "1", head]).split()
        if len(parent) > 1:
            base = f"{head}^"
    if base:
        maybe_fetch(base)
        maybe_fetch(head)
        merge_base = (
            run_git(["merge-base", base, head]) if ref_exists(base) and ref_exists(head) else ""
        )
        left = merge_base or base
        out = run_git(["diff", "--name-only", f"{left}...{head}"])
        if not out:
            out = run_git(["diff", "--name-only", f"{left}", head])
    else:
        out = run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", head])
        if not out:
            out = run_git(["ls-files"])
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def is_doc(path: str) -> bool:
    p = Path(path)
    return (
        path in DOC_FILES
        or path.startswith(DOC_DIRS)
        or path.startswith(DOC_ASSET_DIRS)
        or p.suffix.lower() in DOC_EXTS
    )


def any_path(files: Iterable[str], pred) -> bool:
    return any(pred(f) for f in files)


def classify(files: list[str], event_name: str) -> dict[str, object]:
    workflow_changed = any_path(
        files, lambda f: f.startswith(".github/workflows/") or f in WORKFLOW_FILES
    )
    full_ci_required = event_name in {"workflow_dispatch", "push"} or workflow_changed
    backend_changed = any_path(
        files,
        lambda f: f.startswith("backend/")
        or f.startswith("scripts/")
        or f in {"pyproject.toml", "mypy.ini", "Dockerfile"},
    )
    electron_changed = any_path(
        files,
        lambda f: f.startswith("electron-app/")
        or f.startswith("frontend/")
        or f in {"package.json", "package-lock.json"},
    )
    docker_changed = any_path(
        files,
        lambda f: f in {"Dockerfile", "docker-compose.yml", ".dockerignore"}
        or f.startswith("backend/"),
    )
    python_tooling_changed = any_path(
        files, lambda f: f in {"pyproject.toml", "mypy.ini"} or f.startswith("backend/requirements")
    )
    node_tooling_changed = any_path(
        files,
        lambda f: f in {"package.json", "package-lock.json"}
        or f in {"electron-app/package.json", "electron-app/package-lock.json"},
    )
    docs_changed = any_path(files, is_doc)
    docs_only = bool(files) and all(is_doc(f) for f in files)
    return {
        "files": files,
        "docs_changed": docs_changed,
        "docs_only": docs_only,
        "backend_changed": backend_changed,
        "electron_changed": electron_changed,
        "docker_changed": docker_changed,
        "workflow_changed": workflow_changed,
        "python_tooling_changed": python_tooling_changed,
        "node_tooling_changed": node_tooling_changed,
        "full_ci_required": full_ci_required,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--event", default=os.getenv("GITHUB_EVENT_NAME", "local"))
    parser.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT"))
    args = parser.parse_args()

    base = args.base or os.getenv("GITHUB_BASE_REF")
    if base and not base.startswith("origin/") and not ref_exists(base):
        base = f"origin/{base}"
    head = args.head or os.getenv("GITHUB_SHA") or "HEAD"
    if args.event == "workflow_dispatch":
        files = run_git(["ls-files"]).splitlines()
    else:
        files = changed_between(base, head)
    result = classify(files, args.event)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fh:
            for key, value in result.items():
                if isinstance(value, bool):
                    fh.write(f"{key}={str(value).lower()}\n")
                elif key == "files":
                    fh.write(f"files_json={json.dumps(value)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
