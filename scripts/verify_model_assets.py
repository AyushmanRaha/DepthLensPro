#!/usr/bin/env python3
"""Compatibility wrapper for scripts/manage_model_assets.py verify."""
from __future__ import annotations
import runpy, sys
sys.argv = [sys.argv[0], "verify", *sys.argv[1:]]
runpy.run_path("scripts/manage_model_assets.py", run_name="__main__")
