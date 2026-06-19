#!/usr/bin/env python3
"""Flexible policy checks for the dynamic CI workflow."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKFLOW = Path(".github/workflows/ci.yml")
REQUIRED_JOBS = [
    "detect-changes",
    "backend-quality",
    "electron-contract",
    "docker-build",
    "workflow-policy",
    "docs-contract",
    "ci-passed",
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    require(
        re.search(r"^name:\s*CI\s*$", text, re.M) is not None, "workflow name must be exactly CI"
    )
    require(
        "pull_request:" in text
        and re.search(r"pull_request:[\s\S]*?branches:[\s\S]*?- main", text),
        "pull_request must target main",
    )
    require(
        re.search(r"push:[\s\S]*?branches:[\s\S]*?- main", text), "push must be limited to main"
    )
    require("workflow_dispatch:" in text, "workflow_dispatch trigger is required")
    require(
        "**" not in re.search(r"push:[\s\S]*?(?=\n[a-zA-Z_]+:|\npermissions:)", text).group(0),
        "push must not include **",
    )
    require(
        re.search(r"permissions:\s*\n\s*contents:\s*read", text),
        "permissions must be contents: read",
    )
    require(
        "github.event_name" in text and "github.head_ref" in text and "github.ref_name" in text,
        "concurrency must be event-aware and PR-head-aware",
    )
    require("continue-on-error: true" not in text, "continue-on-error true is forbidden")
    for job in REQUIRED_JOBS:
        require(
            re.search(rf"^\s{{2}}{re.escape(job)}:\s*$", text, re.M), f"missing required job {job}"
        )
    require(
        re.search(r"ci-passed:[\s\S]*?if:\s*\$\{\{\s*always\(\)\s*\}\}", text),
        "ci-passed must use if: always()",
    )
    require("docker/build-push-action@v6" in text, "Docker CI must use docker/build-push-action@v6")
    require("docker build" not in text, "workflow must not use raw docker build")
    require("docker/setup-buildx-action@v3" in text, "Docker CI must set up Buildx")
    for output in [
        "backend_changed",
        "electron_changed",
        "docker_changed",
        "workflow_changed",
        "full_ci_required",
        "docs_only",
    ]:
        require(
            f"needs.detect-changes.outputs.{output}" in text,
            f"detect-changes output {output} must be used",
        )
    for job in [
        "backend-quality",
        "electron-contract",
        "docker-build",
        "workflow-policy",
        "docs-contract",
    ]:
        require(
            re.search(rf"{job}:[\s\S]*?if:\s*\$\{{\{{[\s\S]*?needs\.detect-changes\.outputs", text),
            f"{job} must be dynamically gated by detect-changes",
        )
    require(
        "required_jobs" in text and "::error::" in text and "skipped" in text,
        "ci-passed must validate failed/cancelled/skipped required jobs with annotations",
    )
    print("Workflow policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
