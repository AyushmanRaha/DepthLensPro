#!/usr/bin/env python3
"""Validate required CI workflow policy so branch protection stays reliable."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is not installed
    yaml = None

WORKFLOW = Path(".github/workflows/ci.yml")
REQUIRED_JOBS = ["backend-quality", "electron-contract", "docker-build", "ci-passed"]
AGGREGATE_NEEDS = ["backend-quality", "electron-contract", "docker-build"]
CONCURRENCY_GROUP = (
    "ci-${{ github.workflow }}-"
    "${{ github.event.pull_request.head.repo.full_name || github.repository }}-"
    "${{ github.head_ref || github.ref_name }}"
)


def fail(message: str) -> None:
    print(f"workflow validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "{}"}:
        return {} if value == "{}" else ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            item_text = line[2:]
            if not isinstance(parent, list):
                fail("fallback YAML parser encountered a list outside a list container")
            if ":" in item_text and not item_text.startswith('"'):
                key, value = item_text.split(":", 1)
                item: dict[str, Any] = {key.strip(): parse_scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(parse_scalar(item_text))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {"|", ">"}:
            block_indent = None
            block: list[str] = []
            while index < len(lines):
                peek = lines[index]
                peek_indent = len(peek) - len(peek.lstrip(" "))
                if peek.strip() and peek_indent <= indent:
                    break
                index += 1
                if block_indent is None and peek.strip():
                    block_indent = peek_indent
                block.append(peek[(block_indent or 0) :])
            parsed: Any = "\n".join(block)
        elif value:
            parsed = parse_scalar(value)
        else:
            # Pick list if the next meaningful line at a deeper indent starts with '-'.
            parsed = {}
            for peek in lines[index:]:
                if not peek.strip() or peek.lstrip().startswith("#"):
                    continue
                peek_indent = len(peek) - len(peek.lstrip(" "))
                if peek_indent <= indent:
                    break
                parsed = [] if peek.strip().startswith("- ") else {}
                break
        if isinstance(parent, dict):
            parent[key] = parsed
        else:
            fail("fallback YAML parser encountered a mapping inside a scalar list")
        if isinstance(parsed, (dict, list)):
            stack.append((indent, parsed))
    return root


def load_workflow() -> dict[str, Any]:
    try:
        text = WORKFLOW.read_text(encoding="utf-8")
        data = (
            yaml.load(text, Loader=yaml.BaseLoader) if yaml is not None else minimal_yaml_load(text)
        )
    except Exception as exc:  # noqa: BLE001 - report parser failures clearly
        fail(f"{WORKFLOW} does not parse as YAML: {exc}")
    if not isinstance(data, dict):
        fail(f"{WORKFLOW} must contain a YAML mapping")
    return data


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def validate_triggers(data: dict[str, Any]) -> None:
    triggers = data.get("on")
    if not isinstance(triggers, dict):
        fail("workflow must define explicit 'on' triggers")
    for trigger in ("push", "pull_request"):
        config = triggers.get(trigger)
        if config is None:
            fail(f"workflow trigger '{trigger}' is required")
        if not isinstance(config, dict):
            fail(f"workflow trigger '{trigger}' must be explicit mapping")
        for key in ("paths", "paths-ignore"):
            if key in config:
                fail(f"required trigger '{trigger}' must not use {key}")
    if "workflow_dispatch" not in triggers:
        fail("workflow_dispatch trigger is required")


def validate_permissions(data: dict[str, Any]) -> None:
    if data.get("permissions") != {"contents": "read"}:
        fail("permissions must exactly equal {contents: read}")


def validate_concurrency(data: dict[str, Any]) -> None:
    concurrency = data.get("concurrency")
    if not isinstance(concurrency, dict):
        fail("workflow concurrency mapping is required")
    if concurrency.get("group") != CONCURRENCY_GROUP:
        fail("workflow must use the PR-aware concurrency group")
    if concurrency.get("cancel-in-progress") != "true":
        fail("workflow concurrency cancel-in-progress must be true")


def step_run_contains(job: dict[str, Any], text: str) -> bool:
    return any(
        isinstance(step, dict) and text in str(step.get("run", "")) for step in job.get("steps", [])
    )


def validate_jobs(data: dict[str, Any]) -> None:
    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        fail("jobs mapping is required")
    for job_id in REQUIRED_JOBS:
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            fail(f"required job missing: {job_id}")
        if job.get("name") != job_id:
            fail(f"required job {job_id} display name must exactly match job ID")
        if "timeout-minutes" not in job:
            fail(f"required job {job_id} must set timeout-minutes")
        if job_id != "ci-passed" and "if" in job:
            fail(f"required job {job_id} must not have a job-level if condition")

    aggregate = jobs["ci-passed"]
    if as_list(aggregate.get("needs")) != AGGREGATE_NEEDS:
        fail("ci-passed needs must be exactly backend-quality, electron-contract, docker-build")
    if aggregate.get("if") != "always()":
        fail("ci-passed must use if: always()")

    for job_id, job in jobs.items():
        if isinstance(job, dict) and str(job.get("continue-on-error", "false")).lower() == "true":
            fail(f"job {job_id} must not use continue-on-error: true")
        for index, step in enumerate(job.get("steps", []), start=1):
            if not isinstance(step, dict):
                continue
            if str(step.get("continue-on-error", "false")).lower() == "true":
                fail(f"job {job_id} step {index} must not use continue-on-error: true")

    critical_steps = {
        "backend-quality": ["Install backend dependencies", "Run backend quality gates"],
        "electron-contract": ["Install Electron dependencies", "Run Electron contract gates"],
    }
    for job_id, names in critical_steps.items():
        steps_by_name = {
            step.get("name"): step
            for step in jobs[job_id].get("steps", [])
            if isinstance(step, dict)
        }
        for name in names:
            if name not in steps_by_name:
                fail(f"job {job_id} missing critical step '{name}'")
            if "timeout-minutes" not in steps_by_name[name]:
                fail(f"job {job_id} critical step '{name}' must set timeout-minutes")

    docker_steps = jobs["docker-build"].get("steps", [])
    if not any(
        isinstance(step, dict) and step.get("uses") == "docker/build-push-action@v6"
        for step in docker_steps
    ):
        fail("docker-build must use docker/build-push-action@v6")
    if any(
        isinstance(step, dict) and "docker build" in str(step.get("run", ""))
        for step in docker_steps
    ):
        fail("docker-build workflow must not use plain docker build")
    for step in docker_steps:
        if (
            isinstance(step, dict)
            and step.get("uses") == "docker/build-push-action@v6"
            and "timeout-minutes" not in step
        ):
            fail("Docker build action step must set timeout-minutes")

    for job_id in ("backend-quality", "electron-contract"):
        if not step_run_contains(jobs[job_id], f"scripts/ci.sh {job_id}"):
            fail(f"{job_id} must call scripts/ci.sh {job_id}")


def main() -> None:
    data = load_workflow()
    validate_triggers(data)
    validate_permissions(data)
    validate_concurrency(data)
    validate_jobs(data)
    print("workflow validation passed")


if __name__ == "__main__":
    main()
