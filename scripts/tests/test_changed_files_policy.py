from scripts.changed_files import classify


def flags(paths):
    return classify(paths, "pull_request")


def test_release_affecting_paths_trigger_expected_gates():
    cases = {
        "electron-app/package-lock.json": ("electron_changed", "node_tooling_changed"),
        "backend/requirements-dev.txt": ("backend_changed", "python_tooling_changed"),
        "scripts/setup-macos.sh": ("backend_changed", "electron_changed"),
        "frontend/js/api-client.js": ("electron_changed",),
        "Dockerfile": ("backend_changed",),
        ".github/workflows/ci.yml": ("workflow_changed",),
        "docs/screenshots/home.png": ("docs_changed",),
        "SECURITY.md": ("docs_changed", "workflow_changed"),
        ".gitignore": ("docs_changed", "workflow_changed"),
    }
    for path, expected in cases.items():
        result = flags([path])
        for key in expected:
            assert result[key], f"{path} should set {key}"


def test_docs_only_excludes_release_tooling():
    assert flags(["docs/testing-and-ci.md"])["docs_only"] is True
    assert flags(["README.md", "scripts/ci.sh"])["docs_only"] is False
