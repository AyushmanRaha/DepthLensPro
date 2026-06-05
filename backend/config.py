"""Structured runtime configuration for the DepthLens Pro backend."""

from __future__ import annotations

import importlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import Field, field_validator

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings, SettingsConfigDict
elif importlib.util.find_spec("pydantic_settings") is not None:
    pydantic_settings = importlib.import_module("pydantic_settings")
    BaseSettings = cast(Any, pydantic_settings.BaseSettings)
    SettingsConfigDict = cast(Any, pydantic_settings.SettingsConfigDict)
else:
    from pydantic import BaseModel as BaseSettings

    def SettingsConfigDict(**kwargs: Any) -> dict[str, Any]:
        """Compatibility shim when local dev dependencies are not installed yet."""

        return kwargs


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_ENV_FILE = Path(".env")
_ENV_KEYS = ("HOST", "PORT", "LOG_LEVEL", "DEBUG")


class Settings(BaseSettings):  # type: ignore[misc]
    """Environment-backed application settings.

    Values are loaded from process environment variables first and then from a
    local ``.env`` file when present, allowing Docker Compose and local
    development flows to share a single configuration surface. The deployed
    dependency path uses ``pydantic-settings``; the lightweight compatibility
    path keeps local tests importable before dependencies are installed.
    """

    HOST: str = Field(default="0.0.0.0", description="Host interface for the ASGI server.")
    PORT: int = Field(default=8000, ge=1, le=65535, description="ASGI server port.")
    LOG_LEVEL: LogLevel = Field(default="INFO", description="Backend logging level.")
    DEBUG: bool = Field(default=False, description="Enable FastAPI debug responses.")

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.upper()
        return value

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


def _dotenv_values(env_file: Path = _ENV_FILE) -> dict[str, str]:
    """Parse simple KEY=VALUE pairs from a dotenv file for the local fallback path."""

    if not env_file.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip().strip('"').strip("'")
    return values


def _settings_values() -> dict[str, str]:
    values = _dotenv_values()
    values.update({key: os.environ[key] for key in _ENV_KEYS if key in os.environ})
    return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings for deterministic runtime behavior."""

    return Settings(**_settings_values())


settings = get_settings()
