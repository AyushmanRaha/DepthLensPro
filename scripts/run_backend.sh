#!/usr/bin/env bash
set -euo pipefail
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
