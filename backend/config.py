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
_ENV_KEYS = (
    "HOST",
    "PORT",
    "LOG_LEVEL",
    "DEBUG",
    "REDIS_URL",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "REDIS_SOCKET_TIMEOUT_SECONDS",
    "REDIS_MAX_CONNECTIONS",
    "CACHE_TTL_SECONDS",
    "CACHE_MAX_ENTRIES",
    "DEPTHLENS_PRELOAD_MODEL",
    "DEPTHLENS_WARMUP_MODEL",
    "DEPTHLENS_WARMUP_DEVICE",
    "DEPTHLENS_MAX_DIM",
    "DEPTHLENS_DEFAULT_METRICS",
    "DEPTHLENS_DEFAULT_OUTPUTS",
)


class Settings(BaseSettings):
    """Environment-backed application settings.

    Values are loaded from process environment variables first and then from a
    local ``.env`` file when present, allowing Docker Compose and local
    development flows to share a single configuration surface. The deployed
    dependency path uses ``pydantic-settings``; the lightweight compatibility
    path keeps local tests importable before dependencies are installed.
    """

    HOST: str = Field(default="127.0.0.1", description="Host interface for the ASGI server.")
    PORT: int = Field(default=8765, ge=1, le=65535, description="ASGI server port.")
    LOG_LEVEL: LogLevel = Field(default="INFO", description="Backend logging level.")
    DEBUG: bool = Field(default=False, description="Enable FastAPI debug responses.")
    REDIS_URL: str | None = Field(default=None, description="Full Redis connection URL override.")
    REDIS_HOST: str = Field(default="127.0.0.1", description="Redis cache host.")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535, description="Redis cache port.")
    REDIS_DB: int = Field(default=0, ge=0, description="Redis logical database index.")
    REDIS_PASSWORD: str | None = Field(default=None, description="Optional Redis password.")
    REDIS_SOCKET_TIMEOUT_SECONDS: float = Field(
        default=1.5, gt=0, description="Redis socket connect/read timeout in seconds."
    )
    REDIS_MAX_CONNECTIONS: int = Field(
        default=20, ge=1, description="Maximum Redis connections in the shared pool."
    )
    CACHE_TTL_SECONDS: int = Field(
        default=3600, ge=1, description="TTL applied to generated cache keys."
    )
    CACHE_MAX_ENTRIES: int = Field(
        default=256, ge=1, description="Maximum entries retained by the in-memory cache."
    )
    DEPTHLENS_PRELOAD_MODEL: bool = Field(
        default=False,
        description="Run optional model warmup in the background after FastAPI is live.",
    )
    DEPTHLENS_WARMUP_MODEL: str = Field(
        default="MiDaS_small", description="Model to warm when DEPTHLENS_PRELOAD_MODEL is enabled."
    )
    DEPTHLENS_WARMUP_DEVICE: str = Field(
        default="auto", description="Device to warm when DEPTHLENS_PRELOAD_MODEL is enabled."
    )
    DEPTHLENS_MAX_DIM: int = Field(
        default=1536,
        ge=256,
        le=4096,
        description="Maximum long image edge for interactive inference.",
    )
    DEPTHLENS_DEFAULT_METRICS: str = Field(
        default="fast", description="Default estimate metrics mode: none, fast, or full."
    )
    DEPTHLENS_DEFAULT_OUTPUTS: str = Field(
        default="color", description="Default estimate outputs: color, gray, or color,gray."
    )

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

    return Settings(**cast(Any, _settings_values()))


settings = get_settings()
