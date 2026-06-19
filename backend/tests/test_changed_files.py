from __future__ import annotations

from scripts.changed_files import classify


def test_classify_readme_only_as_docs_only() -> None:
    result = classify(["README.md"], "pull_request")
    assert result["docs_only"] is True
    assert result["docs_changed"] is True
    assert result["backend_changed"] is False
    assert result["electron_changed"] is False
    assert result["docker_changed"] is False
    assert result["full_ci_required"] is False


def test_classify_backend_change_requires_backend_and_docker() -> None:
    result = classify(["backend/app.py"], "pull_request")
    assert result["backend_changed"] is True
    assert result["docker_changed"] is True
    assert result["docs_only"] is False


def test_classify_workflow_change_requires_full_ci() -> None:
    result = classify([".github/workflows/ci.yml"], "pull_request")
    assert result["workflow_changed"] is True
    assert result["full_ci_required"] is True


def test_classify_push_conservatively_requires_full_ci() -> None:
    result = classify(["README.md"], "push")
    assert result["full_ci_required"] is True
