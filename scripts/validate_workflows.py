"""Validate CI workflow invariants that protect required checks.

This intentionally checks repository policy rather than GitHub's full workflow
schema. It prevents common edits that make required status checks disappear or
let aggregate jobs report success after a dependency was skipped or failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
REQUIRED_JOBS = ("backend-quality", "electron-contract", "docker-build")
AGGREGATE_JOB = "ci-passed"


def fail(message: str) -> None:
    raise SystemExit(f"workflow validation failed: {message}")


def load_workflow() -> dict[str, Any]:
    with CI_WORKFLOW.open("r", encoding="utf-8") as handle:
        workflow = yaml.load(handle, Loader=yaml.BaseLoader)
    if not isinstance(workflow, dict):
        fail(f"{CI_WORKFLOW} did not parse to a mapping")
    return workflow


def as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    fail(f"expected a string or list of strings, got {value!r}")


def main() -> None:
    workflow = load_workflow()

    triggers = workflow.get("on")
    if not isinstance(triggers, dict):
        fail("workflow must define explicit push and pull_request triggers")
    for event in ("push", "pull_request"):
        if event not in triggers:
            fail(f"missing {event} trigger")
        event_config = triggers[event]
        if isinstance(event_config, dict):
            forbidden = {"paths", "paths-ignore"}.intersection(event_config)
            if forbidden:
                fail(
                    f"{event} uses {', '.join(sorted(forbidden))}; "
                    "required checks must not be path-filtered"
                )

    permissions = workflow.get("permissions")
    if permissions != {"contents": "read"}:
        fail("workflow permissions must remain least-privilege: contents: read")

    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        fail("workflow must define jobs")

    for job_id in (*REQUIRED_JOBS, AGGREGATE_JOB):
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            fail(f"missing required job {job_id}")
        if job.get("name") != job_id:
            fail(f"job {job_id} must keep stable display name {job_id!r}")
        if "if" in job and job_id != AGGREGATE_JOB:
            fail(f"required job {job_id} must not have a job-level if condition")

    aggregate = jobs[AGGREGATE_JOB]
    needs = set(as_list(aggregate.get("needs")))
    if needs != set(REQUIRED_JOBS):
        fail(f"{AGGREGATE_JOB} must need exactly {', '.join(REQUIRED_JOBS)}")
    if aggregate.get("if") != "always()":
        fail(f"{AGGREGATE_JOB} must use if: always() so it reports dependency failures")

    for job_id, job in jobs.items():
        if job.get("continue-on-error") == "true":
            fail(f"job {job_id} must not use continue-on-error")
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            fail(f"job {job_id} steps must be a list")
        for index, step in enumerate(steps, start=1):
            if isinstance(step, dict) and step.get("continue-on-error") == "true":
                fail(f"job {job_id} step {index} must not use continue-on-error")

    print("workflow validation passed")


if __name__ == "__main__":
    main()
