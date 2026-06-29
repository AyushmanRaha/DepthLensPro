"""Import-light benchmark configuration."""

from __future__ import annotations

import os

BENCHMARK_TIMEOUT_SECONDS = int(os.getenv("DEPTHLENS_BENCHMARK_TIMEOUT_SECONDS", "180"))
