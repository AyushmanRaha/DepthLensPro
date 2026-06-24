#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess


def run_git(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], text=True, capture_output=True, check=check)


def git_lines(args: list[str]) -> list[str]:
    proc = run_git(args)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_ref(ref: str) -> None:
    if not ref:
        return
    ref_name = ref.removeprefix("refs/heads/").removeprefix("origin/")
    run_git(
        [
            "fetch",
            "--no-tags",
            "--prune",
            "--depth=100",
            "origin",
            f"{ref_name}:refs/remotes/origin/{ref_name}",
        ]
    )


def merge_base(base: str, head: str) -> str | None:
    proc = run_git(["merge-base", base, head])
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def diff_files(base: str, head: str) -> list[str]:
    return git_lines(["diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...{head}"])


def _matches_any(path: str, prefixes: tuple[str, ...] = (), exact: set[str] | None = None) -> bool:
    return path in (exact or set()) or any(path.startswith(prefix) for prefix in prefixes)


def classify(files: list[str], event: str) -> dict[str, bool]:
    docs = backend = electron = workflow = pytool = nodetool = False
    for raw in files:
        s = raw.replace("\\", "/")
        is_doc = (
            s in {"README.md", "CONTRIBUTING.md", "LICENSE", "SECURITY.md"}
            or s.startswith(("docs/", "images/screenshots/", "assets/architecture/"))
            or s.endswith(".md")
        )
        docs = docs or is_doc

        if _matches_any(
            s,
            prefixes=("backend/", "backend/scripts/"),
            exact={"pyproject.toml", "mypy.ini", "Dockerfile", "docker-compose.yml"},
        ):
            backend = True
        if s.startswith("backend/requirements") or s in {"pyproject.toml", "mypy.ini"}:
            pytool = True
        if _matches_any(
            s,
            prefixes=("electron-app/", "frontend/"),
            exact={"package.json", "package-lock.json"},
        ):
            electron = True
        if s in {
            "package.json",
            "package-lock.json",
            "electron-app/package.json",
            "electron-app/package-lock.json",
        }:
            nodetool = True
        if _matches_any(
            s,
            prefixes=(".github/workflows/",),
            exact={
                "scripts/ci.sh",
                "scripts/changed_files.py",
                "scripts/validate_workflows.py",
                "scripts/validate_docs.py",
            },
        ):
            workflow = True
        if _matches_any(
            s,
            prefixes=("scripts/setup-", "scripts/build-", "scripts/launch-", "scripts/prefetch-"),
            exact={
                "scripts/doctor.py",
                "scripts/setup_state.py",
                "scripts/diagnose_backend.py",
            },
        ):
            backend = True
            electron = True
        if s in {".gitignore", "SECURITY.md"}:
            docs = True
            workflow = True
        if s.startswith(("docs/resource-path-contract", "resource-policy", "models/")):
            backend = True
            electron = True
            docs = True

    full = event in {"push", "workflow_dispatch"}
    non_docs = [
        f
        for f in files
        if not (
            f.endswith(".md")
            or f in {"LICENSE"}
            or f.startswith(("docs/", "images/screenshots/", "assets/architecture/"))
        )
    ]
    return {
        "docs_changed": docs,
        "docs_only": bool(files) and docs and not non_docs,
        "backend_changed": backend,
        "electron_changed": electron,
        "workflow_changed": workflow,
        "python_tooling_changed": pytool,
        "node_tooling_changed": nodetool,
        "full_ci_required": full,
    }


def determine_files(args: argparse.Namespace) -> tuple[list[str], bool, str]:
    event = os.getenv("GITHUB_EVENT_NAME", "local")
    head = args.head or os.getenv("GITHUB_SHA") or "HEAD"
    base = args.base
    uncertain = False
    reason = ""
    if event == "pull_request":
        base_ref = base or os.getenv("GITHUB_BASE_REF", "main")
        fetch_ref(base_ref)
        base_candidate = f"origin/{base_ref.removeprefix('refs/heads/')}"
        mb = merge_base(base_candidate, head) or merge_base(base_ref, head)
        if not mb:
            uncertain = True
            reason = f"could not find merge-base for {base_ref}..{head}; using HEAD^ fallback"
            mb = "HEAD^"
        files = diff_files(mb, head) or git_lines(["diff", "--name-only", f"{mb}", head])
        return files, uncertain, reason or f"pull_request diff {mb}...{head}"
    if base:
        mb = merge_base(base, head) or base
        return diff_files(mb, head), False, f"explicit diff {mb}...{head}"
    if event == "push":
        before = os.getenv("GITHUB_EVENT_BEFORE") or os.getenv("GITHUB_BEFORE")
        if before and set(before) != {"0"}:
            files = git_lines(["diff", "--name-only", "--diff-filter=ACMRTUXB", before, head])
            return files, False, f"push diff {before}..{head}"
        uncertain = True
        return [], uncertain, "push event without usable before SHA; full CI required"
    return [], True, f"{event} event; full CI required"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    event = os.getenv("GITHUB_EVENT_NAME", "local")
    files, uncertain, reason = determine_files(args)
    outputs = classify(files, event)
    if uncertain:
        outputs["full_ci_required"] = True
    payload = {"changed_files": files, "reason": reason, **outputs}
    print(f"Changed-file detection: {reason}")
    print(f"Changed files ({len(files)}):")
    for f in files:
        print(f"  - {f}")
    print("Classification:")
    for k, v in outputs.items():
        print(f"  {k}={str(v).lower()}")
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            for k, v in outputs.items():
                fh.write(f"{k}={str(v).lower()}\n")
            fh.write("changed_files<<EOF\n" + "\n".join(files) + "\nEOF\n")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if event == "pull_request" and uncertain and not files:
        print(
            "::warning::Pull request diff was uncertain and empty; "
            "workflow changes should be inspected."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
