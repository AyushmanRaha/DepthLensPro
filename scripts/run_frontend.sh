#!/usr/bin/env bash
set -euo pipefail
python3 -m http.server 5500 --directory frontend
