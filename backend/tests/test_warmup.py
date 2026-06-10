from __future__ import annotations

import pytest

from backend import main


@pytest.mark.parametrize("env_name", ["TESTING", "DEPTHLENS_SKIP_WARMUP"])
def test_warmup_respects_testing_and_skip_env(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    import asyncio

    monkeypatch.setenv(env_name, "1")

    def fail_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("warmup should be skipped")

    monkeypatch.setattr("backend.services.inference._load_model", fail_load)

    asyncio.run(main._warm_default_model())
