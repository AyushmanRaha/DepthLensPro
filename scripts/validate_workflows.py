#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def fallback_parse(text: str) -> dict:
    # Minimal CI-policy parser for environments without PyYAML. It intentionally
    # extracts only the durable contracts this validator checks.
    data: dict = {"jobs": {}}
    lines = text.splitlines()
    for line in lines:
        if line.startswith("name:"):
            data["name"] = line.split(":", 1)[1].strip()
    data["on"] = {
        "push": {"branches": ["main"]} if "  push:" in text and "      - main" in text else {},
        "pull_request": (
            {"branches": ["main"]} if "  pull_request:" in text and "      - main" in text else {}
        ),
    }
    if "  workflow_dispatch:" in text:
        data["on"]["workflow_dispatch"] = None
    if "permissions:\n  contents: read" in text:
        data["permissions"] = {"contents": "read"}
    group = ""
    for i, line in enumerate(lines):
        if line.strip() == "group:" or line.strip().startswith("group:"):
            group = line
            break
    data["concurrency"] = {"group": group} if group else {}
    current = None
    in_jobs = False
    for line in lines:
        if line == "jobs:":
            in_jobs = True
            continue
        if (
            in_jobs
            and line.startswith("  ")
            and not line.startswith("    ")
            and line.strip().endswith(":")
        ):
            current = line.strip()[:-1]
            data["jobs"][current] = {"raw": ""}
        elif current:
            data["jobs"][current]["raw"] += line + "\n"
            if line.strip().startswith("if:"):
                data["jobs"][current]["if"] = line.split(":", 1)[1].strip()
            if line.strip() == "needs:":
                data["jobs"][current]["needs"] = []
            elif "needs" in data["jobs"][current] and line.strip().startswith("- "):
                data["jobs"][current]["needs"].append(line.strip()[2:])
    return data


WORKFLOW = Path(".github/workflows/ci.yml")
REQUIRED_JOBS = [
    "detect-changes",
    "workflow-policy",
    "docs-contract",
    "backend-quality",
    "electron-contract",
    "ci-passed",
]


def err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def contains(obj, needle: str) -> bool:
    return needle in str(obj)


def main() -> int:
    errors: list[str] = []
    if not WORKFLOW.is_file():
        print("::error::.github/workflows/ci.yml is missing")
        return 1
    text = WORKFLOW.read_text(encoding="utf-8")
    if yaml is None:
        data = fallback_parse(text)
    else:

        class Loader(yaml.SafeLoader):
            pass

        Loader.add_constructor(
            "tag:yaml.org,2002:bool", lambda loader, node: loader.construct_scalar(node)
        )
        try:
            data = yaml.load(text, Loader=Loader) or {}
        except Exception as exc:
            err(errors, f"workflow YAML does not parse: {exc}")
            data = {}
    if data.get("name") != "CI":
        err(errors, "workflow name must be CI")
    on = data.get("on", {})
    if on.get("push", {}).get("branches") != ["main"]:
        err(errors, "push trigger must target only main")
    if "**" in str(on.get("push", {})):
        err(errors, "push trigger must not include **")
    if on.get("pull_request", {}).get("branches") != ["main"]:
        err(errors, "pull_request trigger must target main")
    if "workflow_dispatch" not in on:
        err(errors, "workflow_dispatch trigger is required")
    if data.get("permissions") != {"contents": "read"}:
        err(errors, "permissions must be contents: read")
    concurrency = data.get("concurrency", {})
    if not all(token in str(concurrency) for token in ["github.workflow", "github.event_name"]):
        err(errors, "concurrency group must include github.workflow and github.event_name")
    if not ("github.head_ref" in str(concurrency) or "github.ref_name" in str(concurrency)):
        err(errors, "concurrency group must include github.head_ref or github.ref_name")
    jobs = data.get("jobs", {})
    for job in REQUIRED_JOBS:
        if job not in jobs:
            err(errors, f"missing required job: {job}")
    if "docker-build" in jobs:
        err(errors, "docker-build must not be a required CI job")
    ci = jobs.get("ci-passed", {})
    if str(ci.get("if", "")).strip() != "always()":
        err(errors, "ci-passed must use if: always()")
    needs = ci.get("needs", [])
    for job in REQUIRED_JOBS[:-1]:
        if job not in needs:
            err(errors, f"ci-passed must need {job}")
    for job_name, job in jobs.items():
        if (
            job.get("continue-on-error") is True
            or str(job.get("continue-on-error")).lower() == "true"
        ):
            err(errors, f"{job_name} must not use continue-on-error: true")
        for i, step in enumerate(job.get("steps", []) or [], start=1):
            if (
                step.get("continue-on-error") is True
                or str(step.get("continue-on-error")).lower() == "true"
            ):
                err(errors, f"{job_name} step {i} must not use continue-on-error: true")
    if not contains(jobs.get("backend-quality", {}), "scripts/ci.sh backend-quality"):
        err(errors, "backend-quality must call scripts/ci.sh backend-quality")
    if not contains(jobs.get("electron-contract", {}), "scripts/ci.sh electron-contract"):
        err(errors, "electron-contract must call scripts/ci.sh electron-contract")
    if not (
        contains(jobs.get("workflow-policy", {}), "scripts/ci.sh workflow-policy")
        or contains(jobs.get("workflow-policy", {}), "scripts/validate_workflows.py")
    ):
        err(
            errors,
            "workflow-policy must run scripts/validate_workflows.py directly "
            "or through scripts/ci.sh",
        )
    if not all(
        token in str(ci)
        for token in [
            "required",
            "skipped",
            "workflow_required",
            "backend_required",
            "electron_required",
        ]
    ):
        err(errors, "ci-passed must distinguish required jobs from acceptable skipped jobs")
    if errors:
        for e in errors:
            print(f"::error::{e}")
        return 1
    print("Workflow policy validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
