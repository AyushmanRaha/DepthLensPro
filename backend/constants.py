"""Shared literal constants for lightweight backend policy.

Keep this module free of imports from heavier backend services so setup scripts
can safely import it before optional runtime dependencies are installed.
"""

from __future__ import annotations

DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8765
MAX_UPLOAD_SIZE_MB = 20

SUPPORTED_METRICS_MODES = ("none", "fast", "full")
SUPPORTED_OUTPUT_MODES = ("color", "gray")
SUPPORTED_ONNX_MODEL_IDS = ("midas_small", "dpt_hybrid", "dpt_large")
CACHE_BACKEND_MEMORY = "memory"
CACHE_BACKEND_REDIS = "redis"
CACHE_BACKEND_MODES = (CACHE_BACKEND_MEMORY, CACHE_BACKEND_REDIS)
RESOURCE_MODE_OFF = "off"
RESOURCE_MODE_OPTIONAL = "optional"
RESOURCE_MODE_REQUIRED = "required"
RESOURCE_MODE_REQUIRE_ALL = "require-all"
CACHE_RESOURCE_MODES = (RESOURCE_MODE_OFF, RESOURCE_MODE_OPTIONAL, RESOURCE_MODE_REQUIRED)
ONNX_RESOURCE_MODES = (*CACHE_RESOURCE_MODES, RESOURCE_MODE_REQUIRE_ALL)
