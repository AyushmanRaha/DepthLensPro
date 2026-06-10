"""Static checks for safe Docker Compose defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_compose_env_file_is_optional_for_fresh_clone() -> None:
    assert "path: .env" in COMPOSE
    assert "required: false" in COMPOSE
    assert "required: true" not in COMPOSE


def test_backend_healthcheck_uses_lightweight_live_endpoint() -> None:
    healthcheck_start = COMPOSE.index("healthcheck:")
    backend_healthcheck = COMPOSE[healthcheck_start : COMPOSE.index("deploy:", healthcheck_start)]
    assert "/live" in backend_healthcheck
    assert "/health" not in backend_healthcheck
