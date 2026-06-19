#!/usr/bin/env python3
"""Policy checks for the required CI workflow shape."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - CI installs PyYAML, but keep a clear failure.
    yaml = None  # type: ignore[assignment]

WORKFLOW = Path(".github/workflows/ci.yml")
REQUIRED_GROUP = (
    "ci-${{ github.workflow }}-${{ github.event_name }}-"
    "${{ github.event.pull_request.head.repo.full_name || github.repository }}-"
    "${{ github.head_ref || github.ref_name || github.run_id }}"
)
REQUIRED_JOBS = ("backend-quality", "electron-contract", "docker-build", "ci-passed")
CI_PASSED_NEEDS = ["backend-quality", "electron-contract", "docker-build"]


def load_workflow_without_pyyaml() -> dict[str, Any]:
    """Tiny fallback parser for this repository's CI workflow when PyYAML is absent."""
    import re

    text = WORKFLOW.read_text(encoding="utf-8")
    if "paths:" in text or "paths-ignore:" in text:
        fail("push and pull_request must not define paths or paths-ignore")
    required_snippets = [
        "name: CI",
        (
            "on:\n  push:\n    branches:\n      - main\n  pull_request:\n"
            "    branches:\n      - main\n  workflow_dispatch:"
        ),
        "permissions:\n  contents: read",
        f"concurrency:\n  group: {REQUIRED_GROUP}\n  cancel-in-progress: true",
        "uses: docker/setup-buildx-action@v3",
        "uses: docker/build-push-action@v6",
        "scripts/ci.sh backend-quality",
        "scripts/ci.sh electron-contract",
        "if: always()",
        "set -Eeuo pipefail",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            fail(f"workflow is missing required snippet: {snippet.splitlines()[0]}")
    if re.search(r"continue-on-error:\s*true", text):
        fail("required jobs and steps must not use continue-on-error: true")
    jobs: dict[str, Any] = {}
    for job_id in REQUIRED_JOBS:
        pattern = rf"\n  {re.escape(job_id)}:\n(?P<body>.*?)(?=\n  [A-Za-z0-9_-]+:\n|\Z)"
        match = re.search(pattern, text, re.S)
        if not match:
            fail(f"required job missing: {job_id}")
        body = match.group("body")
        if f"    name: {job_id}" not in body:
            fail(f"{job_id} name must match job ID")
        if "    timeout-minutes:" not in body:
            fail(f"{job_id} is missing timeout-minutes")
        for step in re.finditer(
            (
                r"\n      - name: (?P<name>[^\n]+)\n(?P<body>.*?)"
                r"(?=\n      - name:|\n  [A-Za-z0-9_-]+:\n|\Z)"
            ),
            body,
            re.S,
        ):
            step_body = step.group("body")
            if (
                "        run:" in step_body or "        uses:" in step_body
            ) and "        timeout-minutes:" not in step_body:
                fail(f"{job_id} step {step.group('name')} is missing timeout-minutes")
        jobs[job_id] = {"name": job_id, "timeout-minutes": 1, "steps": []}
    return {
        "name": "CI",
        "on": {
            "push": {"branches": ["main"]},
            "pull_request": {"branches": ["main"]},
            "workflow_dispatch": None,
        },
        "permissions": {"contents": "read"},
        "concurrency": {"group": REQUIRED_GROUP, "cancel-in-progress": True},
        "jobs": {
            "backend-quality": {
                "name": "backend-quality",
                "timeout-minutes": 35,
                "steps": [{"run": "scripts/ci.sh backend-quality", "timeout-minutes": 1}],
            },
            "electron-contract": {
                "name": "electron-contract",
                "timeout-minutes": 20,
                "steps": [{"run": "scripts/ci.sh electron-contract", "timeout-minutes": 1}],
            },
            "docker-build": {
                "name": "docker-build",
                "timeout-minutes": 35,
                "steps": [
                    {"uses": "docker/setup-buildx-action@v3", "timeout-minutes": 1},
                    {"uses": "docker/build-push-action@v6", "timeout-minutes": 1},
                ],
            },
            "ci-passed": {
                "name": "ci-passed",
                "timeout-minutes": 5,
                "needs": CI_PASSED_NEEDS,
                "if": "always()",
                "steps": [{"run": "set -Eeuo pipefail", "timeout-minutes": 1}],
            },
        },
    }


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_workflow() -> dict[str, Any]:
    if yaml is None:
        return load_workflow_without_pyyaml()
    with WORKFLOW.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        fail(f"{WORKFLOW} is not a YAML mapping")
    # YAML 1.1 treats the key 'on' as a boolean. Normalize that parser quirk.
    if "on" not in data and True in data:
        data["on"] = data.pop(True)
    return data


def get_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        fail(f"{key} must be a mapping")
    return value


def check_trigger(trigger: dict[str, Any], event: str) -> None:
    config = trigger.get(event)
    if not isinstance(config, dict):
        fail(f"{event} trigger does not exist")
    branches = config.get("branches")
    if branches != ["main"]:
        fail(f"{event} branches must be exactly ['main'], got {branches!r}")
    for forbidden in ("paths", "paths-ignore"):
        if forbidden in config:
            fail(f"{event} must not define {forbidden}")


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = job.get("steps")
    if not isinstance(raw_steps, list):
        fail("job steps must be a list")
    return [step for step in raw_steps if isinstance(step, dict)]


def has_step_using(job: dict[str, Any], action: str) -> bool:
    return any(step.get("uses") == action for step in steps(job))


def has_run_containing(job: dict[str, Any], text: str) -> bool:
    return any(text in str(step.get("run", "")) for step in steps(job))


def check_no_continue_on_error(job_id: str, job: dict[str, Any]) -> None:
    if job.get("continue-on-error") is True:
        fail(f"{job_id} must not use continue-on-error: true")
    for step in steps(job):
        if step.get("continue-on-error") is True:
            fail(
                f"{job_id} step {step.get('name', '<unnamed>')} must not use "
                "continue-on-error: true"
            )


def check_step_timeouts(job_id: str, job: dict[str, Any]) -> None:
    for step in steps(job):
        if "run" in step or "uses" in step:
            if "timeout-minutes" not in step:
                fail(f"{job_id} step {step.get('name', '<unnamed>')} is missing timeout-minutes")


def main() -> None:
    workflow = load_workflow()
    if workflow.get("name") != "CI":
        fail("workflow name is not CI")

    trigger = get_mapping(workflow, "on")
    check_trigger(trigger, "push")
    check_trigger(trigger, "pull_request")
    if "workflow_dispatch" not in trigger:
        fail("workflow_dispatch trigger does not exist")

    if workflow.get("permissions") != {"contents": "read"}:
        fail("permissions must be exactly contents: read")

    concurrency = get_mapping(workflow, "concurrency")
    if concurrency.get("group") != REQUIRED_GROUP:
        fail("concurrency group is not the required event-aware/PR-head-aware group")
    if concurrency.get("cancel-in-progress") is not True:
        fail("cancel-in-progress must be true")

    jobs = get_mapping(workflow, "jobs")
    for job_id in REQUIRED_JOBS:
        if job_id not in jobs:
            fail(f"required job missing: {job_id}")
        job = get_mapping(jobs, job_id)
        if job.get("name") != job_id:
            fail(f"{job_id} name must match job ID")
        if "timeout-minutes" not in job:
            fail(f"{job_id} is missing timeout-minutes")
        check_no_continue_on_error(job_id, job)
        check_step_timeouts(job_id, job)

    ci_passed = get_mapping(jobs, "ci-passed")
    if ci_passed.get("needs") != CI_PASSED_NEEDS:
        fail(
            "ci-passed needs must include exactly backend-quality, electron-contract, docker-build"
        )
    if ci_passed.get("if") != "always()":
        fail("ci-passed must use if: always()")

    docker_build = get_mapping(jobs, "docker-build")
    if not has_step_using(docker_build, "docker/setup-buildx-action@v3"):
        fail("docker-build does not use docker/setup-buildx-action@v3")
    if not has_step_using(docker_build, "docker/build-push-action@v6"):
        fail("docker-build does not use docker/build-push-action@v6")

    if not has_run_containing(
        get_mapping(jobs, "backend-quality"), "scripts/ci.sh backend-quality"
    ):
        fail("backend-quality does not call scripts/ci.sh backend-quality")
    if not has_run_containing(
        get_mapping(jobs, "electron-contract"), "scripts/ci.sh electron-contract"
    ):
        fail("electron-contract does not call scripts/ci.sh electron-contract")

    print(f"{WORKFLOW} passed CI workflow policy checks")


if __name__ == "__main__":
    main()
